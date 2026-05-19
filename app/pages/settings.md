<|container|

# Settings

## Thresholds

<|layout|columns=1 1 1|gap=20px|

<|part|class_name=stat-card|
**Temperature: max (C)**

<|{threshold_temp_max}|number|>
|>

<|part|class_name=stat-card|
**Temperature: min (C)**

<|{threshold_temp_min}|number|>
|>

<|part|class_name=stat-card|
**Humidity: max (%)**

<|{threshold_humidity_max}|number|>
|>

|>

<br/>

<|layout|columns=1 1 1|gap=20px|

<|part|class_name=stat-card|
**MQ-2: warning**

<|{threshold_mq2_warn}|number|>
|>

<|part|class_name=stat-card|
**MQ-2: alert**

<|{threshold_mq2_alert}|number|>
|>

<|part|class_name=stat-card|
**Current: max (mA)**

<|{threshold_current_max}|number|>
|>

|>

<br/>

<|layout|columns=1 3|gap=20px|

<|part|class_name=stat-card|
**Illuminance: min (lx)**

<|{threshold_lux_min}|number|>
|>

<|part|
<|Save thresholds|button|on_action=on_save_thresholds|>

*Thresholds apply automatically within a minute - no restart required.*

<|{last_error}|text|>
|>

|>

<br/>

## External publication

<|layout|columns=1 1 1|gap=20px|

<|part|class_name=stat-card|
**External API**

<|{external_calls_display}|text|class_name=h3|>

Pinata + Hedera publish only when allowed.
|>

<|part|class_name=stat-card|
**IPFS (Pinata)**

<|{ipfs_provider_display}|text|class_name=h3|>
|>

<|part|class_name=stat-card|
**Hedera HCS**

<|{hedera_enabled_display}|text|class_name=h3|>

topic: <|{hedera_topic_id}|text|>
|>

|>

<br/>

### Publication schedule

<|layout|columns=1 1 1|gap=20px|

<|part|class_name=stat-card|
**Last publish**

<|{hedera_last_publish}|text|>

status: <|{hedera_last_status}|text|>
|>

<|part|class_name=stat-card|
**Next (scheduled)**

<|{hedera_next_publish}|text|>

automatic, every 5 minutes.
|>

<|part|class_name=stat-card|
**Publish now**

<|Run|button|on_action=on_publish_now|>

Without waiting for the scheduler.
|>

|>

<br/>

### Latest proofs

<|{proofs_recent}|table|columns=idempotency_key;pinata_cid;hedera_sequence_number;hedera_consensus_ts|page_size=10|height=320px|width=100%|>

## AI

<|layout|columns=1 1|gap=20px|

<|part|class_name=stat-card|
**AI forecasting**

<|{ai_enabled_display}|text|class_name=h3|>

period: every minute
|>

<|part|class_name=stat-card|
**Current model**

<|{prediction_value_display}|text|class_name=h3|>
|>

|>

|>
