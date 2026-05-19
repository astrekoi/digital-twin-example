<|container|

# Charts

**Quick range:** <|5 min|button|on_action=on_range_5m|class_name=plain|> <|15 min|button|on_action=on_range_15m|class_name=plain|> <|1 hour|button|on_action=on_range_1h|class_name=plain|> <|6 hours|button|on_action=on_range_6h|class_name=plain|> <|24 hours|button|on_action=on_range_24h|class_name=plain|> <|7 days|button|on_action=on_range_7d|class_name=plain|> | <|Refresh|button|on_action=on_refresh_chart|>

<br/>

**Custom range (Grafana-style):**

<|layout|columns=1 1 auto|gap=12px|

<|part|
**From:**
<|{chart_date_from}|date|with_time|>
|>

<|part|
**To:**
<|{chart_date_to}|date|with_time|>
|>

<|part|
<br/>
<|Apply range|button|on_action=on_apply_date_range|class_name=plain|>
|>

|>

*Current range: <|{chart_range_label}|text|raw|> | auto-refresh every 30 s keeps the selection. Zoom/pan inside a panel is preserved.*

<br/>

<|layout|columns=1 1|gap=16px|

<|chart|figure={chart_temp}|height=260px|>

<|chart|figure={chart_humidity}|height=260px|>

<|chart|figure={chart_pressure}|height=260px|>

<|chart|figure={chart_lux}|height=260px|>

<|chart|figure={chart_current}|height=260px|>

<|chart|figure={chart_mq2}|height=260px|>

|>

|>
