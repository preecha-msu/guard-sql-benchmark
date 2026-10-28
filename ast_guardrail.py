# ไฟล์: 02_guard_sql_engine/ast_guardrail.py
import re
import sqlglot
from sqlglot import exp
from typing import Tuple, List, Dict

class ASTGuardrail:
    """
    โมดูลที่ 1 ของ GUARD-SQL:
    ตรวจสอบและซ่อมแซมโครงสร้าง SQL ระดับ Abstract Syntax Tree (AST)
    แก้ปัญหา C1 (3VL Null Trap), C6 (CASE WHEN Order), และ C7 (Dialect IIF/NVL)
    """
    def __init__(self, target_dialect: str = "postgres"):
        self.target_dialect = target_dialect

    def preprocess_proprietary_syntax(self, sql: str) -> Tuple[str, List[str]]:
        """แปลง NVL -> COALESCE ก่อนเข้า Parser เพื่อรองรับทุก Dialect"""
        logs = []
        # แปลง NVL( เป็น COALESCE( แบบไม่สนใจตัวพิมพ์เล็ก-ใหญ่
        if re.search(r"\bNVL\s*\(", sql, flags=re.IGNORECASE):
            sql = re.sub(r"\bNVL\s*\(", "COALESCE(", sql, flags=re.IGNORECASE)
            logs.append("[C7-Fix] Replaced proprietary NVL() with ANSI COALESCE()")
        return sql, logs

    def _fix_not_in_null_trap(self, expression: exp.Expression, logs: List[str]) -> exp.Expression:
        """
        [C1 Guardrail] ค้นหา 'x NOT IN (SELECT y FROM tbl)'
        และฉีด 'WHERE y IS NOT NULL' เข้าไปใน Subquery ชั้นในเพื่อป้องกัน 3VL Empty Set
        """
        for not_node in expression.find_all(exp.Not):
            in_node = not_node.this
            if isinstance(in_node, exp.In):
                # ตรวจสอบว่าภายใน IN(...) เป็น Subquery หรือไม่
                subquery = in_node.args.get("query")
                if subquery and isinstance(subquery, exp.Subquery):
                    select_exp = subquery.this
                    if isinstance(select_exp, exp.Select) and select_exp.expressions:
                        first_proj = select_exp.expressions[0]
                        col_expr = first_proj.this if isinstance(first_proj, exp.Alias) else first_proj
                        col_sql = col_expr.sql()

                        # เช็คว่าใน WHERE ของ Subquery มี IS NOT NULL อยู่แล้วหรือยัง
                        existing_where = select_exp.args.get("where")
                        where_str = existing_where.sql().upper() if existing_where else ""
                        if "IS NOT NULL" not in where_str:
                            select_exp.where(f"{col_sql} IS NOT NULL", copy=False)
                            logs.append(f"[C1-Fix] Injected '{col_sql} IS NOT NULL' into NOT IN subquery")
        return expression

    def _fix_case_when_specificity(self, expression: exp.Expression, logs: List[str]) -> exp.Expression:
        """
        [C6 Guardrail] ตรวจสอบลำดับเงื่อนไขตัวเลขใน CASE WHEN
        หากเรียง >= ค่าต่ำ (เช่น 50000) ไว้ก่อน >= ค่าสูง (เช่น 100000) ให้สลับลำดับใหม่
        """
        for case_node in expression.find_all(exp.Case):
            ifs = case_node.args.get("ifs", [])
            if len(ifs) < 2:
                continue

            # สกัดค่าตัวเลขจากเงื่อนไข >= หรือ > ในแต่ละ WHEN
            branch_thresholds = []
            can_reorder = True
            for idx, if_node in enumerate(ifs):
                cond = if_node.this
                # ดึงโหนดเปรียบเทียบ >= (GTE) หรือ > (GT) ทั้งกรณีเดี่ยวและกรณีเชื่อมด้วย AND
                gte_nodes = list(cond.find_all((exp.GTE, exp.GT)))
                if gte_nodes and isinstance(gte_nodes[0].expression, exp.Literal) and gte_nodes[0].expression.is_number:
                    threshold_val = float(gte_nodes[0].expression.this)
                    # ถ้ามีเงื่อนไขพ่วงด้วย AND (มีความเฉพาะเจาะจงสูงขึ้น) ให้บวกน้ำหนักเพิ่มเล็กน้อย
                    specificity_bonus = 0.001 if isinstance(cond, exp.And) else 0.0
                    branch_thresholds.append((threshold_val + specificity_bonus, idx, if_node))
                else:
                    can_reorder = False
                    break

            if can_reorder and len(branch_thresholds) == len(ifs):
                # ตรวจสอบว่าเรียงจากมากไปน้อย (Most Restrictive First) อยู่แล้วหรือไม่
                current_vals = [b[0] for b in branch_thresholds]
                sorted_vals = sorted(current_vals, reverse=True)
                if current_vals != sorted_vals:
                    sorted_branches = [b[2] for b in sorted(branch_thresholds, key=lambda x: x[0], reverse=True)]
                    case_node.set("ifs", sorted_branches)
                    logs.append(f"[C6-Fix] Reordered CASE WHEN branches by descending specificity: {current_vals} -> {sorted_vals}")

        return expression

    def apply_guardrails(self, raw_sql: str) -> Dict[str, any]:
        """รันกระบวนการตรวจสอบและแปลงโค้ด SQL ทั้งหมด"""
        sql_cleaned, logs = self.preprocess_proprietary_syntax(raw_sql)

        # ตรวจสอบว่ามี IIF หรือไม่ เพื่อเลือก Dialect ขาเข้าให้ sqlglot แปลงเป็น CASE WHEN
        has_iif = bool(re.search(r"\bIIF\s*\(", sql_cleaned, flags=re.IGNORECASE))
        read_dialect = "tsql" if has_iif else self.target_dialect
        if has_iif:
            logs.append("[C7-Fix] Detected T-SQL IIF(); transpiling to ANSI CASE WHEN")

        try:
            parsed = sqlglot.parse_one(sql_cleaned, read=read_dialect)
            parsed = self._fix_not_in_null_trap(parsed, logs)
            parsed = self._fix_case_when_specificity(parsed, logs)
            safe_sql = parsed.sql(dialect=self.target_dialect, pretty=True)
            return {
                "status": "SUCCESS",
                "original_sql": raw_sql.strip(),
                "guarded_sql": safe_sql,
                "modifications": logs
            }
        except Exception as e:
            return {
                "status": "PARSE_ERROR",
                "original_sql": raw_sql.strip(),
                "guarded_sql": raw_sql.strip(),
                "modifications": logs + [f"[AST-Error] {str(e)}"]
            }