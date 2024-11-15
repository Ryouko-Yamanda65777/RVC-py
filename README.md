# RVC-PY

a simple package to convert voice


# how to use


## download models

you can start by downloading the model by


```
download_models

```

## convert with cli

```

convert_song --uploaded_file path/to/song.wav --voice_model voice_model_name --pitch_change 2.0 --output_format mp3 --f0_method rmvpe


```

## or use python code

```
from rvc_py.main import song_cover_pipeline, download_models


uploaded_file = 'path/to/your/input_song.wav'  # Path to your input audio file
voice_model = 'my_voice_model'  # Name of the voice model directory (e.g., 'my_voice_model' located under 'rvc_models')
pitch_change = 1.5  # Adjust pitch change as needed
output_format = 'mp3'  # Output format (either 'mp3' or 'wav')


index_rate = 0.5
filter_radius = 3
rms_mix_rate = 0.25
f0_method = 'rmvpe'
crepe_hop_length = 128
protect = 0.33
f0_min = 50
f0_max = 1100


try:
    output_path = song_cover_pipeline(
        uploaded_file=uploaded_file,
        voice_model=voice_model,
        pitch_change=pitch_change,
        index_rate=index_rate,
        filter_radius=filter_radius,
        rms_mix_rate=rms_mix_rate,
        f0_method=f0_method,
        crepe_hop_length=crepe_hop_length,
        protect=protect,
        output_format=output_format,
        f0_min=f0_min,
        f0_max=f0_max
    )
    print(f"AI cover generated successfully! Saved at: {output_path}")
except Exception as e:
    print(f"Error during conversion: {e}")
```
