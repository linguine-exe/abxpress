SYSTEM_SUMMARY = """
You are a senior data analyst. Write crisp, decision ready summaries for executives.
Only state results supported by the stats below. Be specific, avoid hype.
"""

USER_SUMMARY_TEMPLATE = """
Context:
- Experiment name: {exp_name}
- Metric type: {metric_type}
- Primary metric: {metric_name}
- Control n={n_c}, Treatment n={n_t}
- Control mean={mean_c:.4f}, Treatment mean={mean_t:.4f}
- Absolute lift={abs_lift:.4f}, Relative lift={rel_lift:.2%}
- 95% CI for lift: [{ci_low:.4f}, {ci_high:.4f}]
- p_value={p_value:.4g}, power={power:.2%}

Constraints:
- If results are not significant, emphasize uncertainty
- If sample is underpowered, recommend sample size
- Include 3 concrete next steps

Write:
1) Four sentence executive summary
2) Bullet list of actions for product, creative, and data engineering
"""
