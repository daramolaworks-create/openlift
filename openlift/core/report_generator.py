import re
from typing import Dict, Any
from html import escape


def _md_inline(text: str) -> str:
    """Convert inline Markdown (**bold**, *italic*) to HTML within a plain string."""
    text = escape(text)
    # Bold must be processed before italic to avoid partial matches.
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    return text

class MarkdownReportGenerator:
    def __init__(self, 
                 experiment_name: str, 
                 test_geo: str, 
                 control_geos: list,
                 metrics: Dict[str, Any],
                 evidence: str,
                 economics: Dict[str, Any],
                 recommendation: str,
                 data_quality: Dict[str, Any],
                 decision: Dict[str, Any] = None):
        self.experiment_name = experiment_name
        self.test_geo = test_geo
        self.control_geos = control_geos
        self.metrics = metrics
        self.evidence = evidence
        self.economics = economics
        self.recommendation = recommendation
        self.data_quality = data_quality
        self.decision = decision or {}

    def generate(self) -> str:
        roas = f"{self.economics.get('incremental_roas', 0):.2f}" if self.economics.get('incremental_roas') else "N/A"
        profit = f"{self.economics.get('incremental_profit', 0):.2f}" if self.economics.get('incremental_profit') else "N/A"
        cac = f"{self.economics.get('incremental_cac', 0):.2f}" if self.economics.get('incremental_cac') else "N/A"
        
        md = f"""# Executive Summary: {self.experiment_name}
        
## 1. Conclusion & Recommendation
**Evidence Strength:** {self.evidence}
**Recommendation:** {self.recommendation}

## 2. Key Metrics
- **Test Geo:** {self.test_geo}
- **Control Geos:** {", ".join(self.control_geos[:3])} (+{max(0, len(self.control_geos)-3)} more)
- **Incremental Outcome:** {self.metrics.get('incremental_outcome_mean', 0):.2f}
- **Lift %:** {self.metrics.get('lift_pct_mean', 0):.1%}
- **Probability Positive:** {self.metrics.get('p_positive', 0) * 100:.1f}%

## 3. Economics
- **Incremental ROAS:** {roas}
- **Incremental Profit:** {profit}
- **Incremental CAC:** {cac}

## 4. Data Quality
- **Score:** {self.data_quality.get('score', 'N/A')}/100 ({self.data_quality.get('status', 'Unknown')})
"""
        if self.data_quality.get('warnings'):
            md += "### Warnings\n"
            for w in self.data_quality.get('warnings'):
                md += f"- {w}\n"

        if self.decision.get("limitations"):
            md += "\n## 5. Limitations\n"
            for item in self.decision["limitations"]:
                md += f"- {item}\n"

        if self.decision.get("next_action"):
            md += f"\n## 6. Next Action\n{self.decision['next_action']}\n"
                
        return md

    def generate_html(self) -> str:
        md = self.generate()
        lines = []
        in_list = False
        for raw_line in md.splitlines():
            line = raw_line.strip()
            if not line:
                if in_list:
                    lines.append("</ul>")
                    in_list = False
                continue
            if line.startswith("# "):
                if in_list:
                    lines.append("</ul>")
                    in_list = False
                lines.append(f"<h1>{_md_inline(line[2:])}</h1>")
            elif line.startswith("## "):
                if in_list:
                    lines.append("</ul>")
                    in_list = False
                lines.append(f"<h2>{_md_inline(line[3:])}</h2>")
            elif line.startswith("### "):
                if in_list:
                    lines.append("</ul>")
                    in_list = False
                lines.append(f"<h3>{_md_inline(line[4:])}</h3>")
            elif line.startswith("- "):
                if not in_list:
                    lines.append("<ul>")
                    in_list = True
                lines.append(f"<li>{_md_inline(line[2:])}</li>")
            else:
                if in_list:
                    lines.append("</ul>")
                    in_list = False
                lines.append(f"<p>{_md_inline(line)}</p>")
        if in_list:
            lines.append("</ul>")

        body = self._visual_summary_html() + "\n" + "\n".join(lines)
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape(self.experiment_name)} - OpenLift Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; line-height: 1.55; color: #111; }}
    h1, h2, h3 {{ color: #000; }}
    h1 {{ border-bottom: 4px solid #FFD700; padding-bottom: 12px; }}
    li {{ margin: 6px 0; }}
    .visual-summary {{ background: #f8f9fa; border: 1px solid #ddd; padding: 20px; margin-bottom: 28px; }}
    .metric-row {{ display: grid; grid-template-columns: 180px 1fr 80px; gap: 12px; align-items: center; margin: 12px 0; }}
    .metric-label {{ font-weight: bold; }}
    .metric-bar {{ height: 14px; background: #e5e5e5; border-radius: 4px; overflow: hidden; }}
    .metric-bar span {{ display: block; height: 100%; background: #FFD700; }}
    .metric-value {{ text-align: right; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""

    def _visual_summary_html(self) -> str:
        lift_pct = self.metrics.get("lift_pct_mean", 0)
        p_positive = self.metrics.get("p_positive", 0)
        dq_score = self.data_quality.get("score", 0)
        roas = self.economics.get("incremental_roas")

        rows = [
            ("Lift", lift_pct, "{:.1%}", -0.5, 0.5),
            ("Probability Positive", p_positive, "{:.1%}", 0, 1),
            ("Data Quality", dq_score / 100 if isinstance(dq_score, (int, float)) else 0, "{:.0%}", 0, 1),
        ]
        if roas is not None:
            rows.append(("Incremental ROAS", min(roas / 5, 1), lambda value: f"{roas:.2f}", 0, 1))

        bars = []
        for label, value, fmt, min_value, max_value in rows:
            normalized = (value - min_value) / (max_value - min_value) if max_value != min_value else 0
            width = max(0, min(100, normalized * 100))
            display = fmt(value) if callable(fmt) else fmt.format(value)
            bars.append(
                f"""
                <div class="metric-row">
                  <div class="metric-label">{escape(label)}</div>
                  <div class="metric-bar"><span style="width:{width:.1f}%"></span></div>
                  <div class="metric-value">{escape(display)}</div>
                </div>
                """
            )

        return f"""
<section class="visual-summary">
  <h2>Visual Summary</h2>
  {''.join(bars)}
</section>
"""
