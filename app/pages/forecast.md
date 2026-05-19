<|container|

# Temperature forecast

<|layout|columns=1 1|gap=20px|

<|part|class_name=stat-card|
**Current forecast**

<|{prediction_value_display}|text|class_name=h1|>

<|{prediction_meta}|text|>
|>

<|part|class_name=stat-card|
**Model and refresh**

<|{selected_model}|selector|lov={available_models}|dropdown|>

<|Apply model|button|on_action=on_apply_model|>  <|Refresh forecast|button|on_action=on_refresh_forecast|>

<|{last_error}|text|>
|>

|>

<br/>

## History + forecast

<|chart|figure={forecast_chart}|height=440px|>

*Solid line - temperature over the last hour. Dashed line - forecast with the +/-1 C uncertainty band.*

<br/>

## Recommendations

<|{recommendations}|table|columns=icon;severity;text|page_size=10|height=280px|width=100%|>

*The system does not act automatically - decisions are taken by the operator.*

|>
