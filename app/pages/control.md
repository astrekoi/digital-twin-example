<|container|

# Control

<|layout|columns=1 3|gap=20px|

<|part|class_name=stat-card|
**Emergency stop**

<|🛑 STOP|button|on_action=on_kill_switch|class_name=plain error|>

All relays off, traffic light off.
|>

<|part|class_name=stat-card|
**Last command status**

<|{last_error}|text|>
|>

|>

<br/>

## Relays (4 channels)

<|layout|columns=1 1 1 1|gap=20px|

<|part|class_name=stat-card|
**Relay 1 - fan**

<|{relay_1}|toggle|on_change=on_relay_1|>
|>

<|part|class_name=stat-card|
**Relay 2 - buzzer**

<|{relay_2}|toggle|on_change=on_relay_2|>
|>

<|part|class_name=stat-card|
**Relay 3 - spare**

<|{relay_3}|toggle|on_change=on_relay_3|>
|>

<|part|class_name=stat-card|
**Relay 4 - spare**

<|{relay_4}|toggle|on_change=on_relay_4|>
|>

|>

<br/>

## Servos

<|layout|columns=auto 1 auto|gap=20px|

<|part|
**Channel**

<|{servo_channel}|selector|lov=1;2|dropdown|>
|>

<|part|
**Angle: <|{servo_angle}|text|raw|> deg**

<|{servo_angle}|slider|min=0|max=180|step=1|>
|>

<|Apply|button|on_action=on_servo_apply|>

|>

<br/>

## Traffic light

<|🔴 Red|button|on_action=on_led_red|class_name=plain|>  
<|🟡 Yellow|button|on_action=on_led_yellow|class_name=plain|>  
<|🟢 Green|button|on_action=on_led_green|class_name=plain|>  
<|⚫ Off|button|on_action=on_led_off|class_name=plain|>

|>
