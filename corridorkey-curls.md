# CorridorKey Curl Commands

## Greenscreen Key (minimal)
```bash
curl -X POST https://platform.indianetailer.in/v1/inference \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"greenscreen_key","video_url":"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4"}'
```

## Greenscreen Key (all params)
```bash
curl -X POST https://platform.indianetailer.in/v1/inference \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"greenscreen_key","video_url":"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4","corridorkey_params":{"use_gvm_alpha":true,"input_is_linear":false,"despill_strength":0.5,"auto_despeckle":true,"despeckle_size":400,"refiner_scale":1.0,"max_frames":0}}'
```

## Background Replace (image bg)
```bash
curl -X POST https://platform.indianetailer.in/v1/inference \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"background_replace","video_url":"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4","image_url":"https://picsum.photos/id/15/1280/720"}'
```

## Background Replace (solid color)
```bash
curl -X POST https://platform.indianetailer.in/v1/inference \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"background_replace","video_url":"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4","corridorkey_params":{"bg_color":"#FF0000"}}'
```

## Background Replace (all params + keep mattes)
```bash
curl -X POST https://platform.indianetailer.in/v1/inference \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"background_replace","video_url":"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4","image_url":"https://picsum.photos/id/15/1280/720","corridorkey_params":{"bg_color":"#000000","gvm_batch":4,"gvm_ensemble":1,"gvm_overlap":2,"refine_threshold":128,"refine_erode":2,"refine_dilate":3,"refine_blur":5,"refine_min_area":500,"refine_temporal_smooth":3,"keep_mattes":true}}'
```

## Rotoscope (minimal)
```bash
curl -X POST https://platform.indianetailer.in/v1/inference \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"rotoscope","video_url":"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4"}'
```

## Rotoscope (matte only)
```bash
curl -X POST https://platform.indianetailer.in/v1/inference \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"rotoscope","video_url":"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4","corridorkey_params":{"export_transparent_pngs":false,"export_subject_video":false,"export_matte_video":true}}'
```

## Rotoscope (subject only, white bg)
```bash
curl -X POST https://platform.indianetailer.in/v1/inference \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"rotoscope","video_url":"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4","corridorkey_params":{"export_transparent_pngs":false,"export_subject_video":true,"export_matte_video":false,"subject_bg_color":"#FFFFFF"}}'
```

## Rotoscope (all params)
```bash
curl -X POST https://platform.indianetailer.in/v1/inference \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed" \
  -H "Content-Type: application/json" \
  -d '{"pipeline":"rotoscope","video_url":"https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4","corridorkey_params":{"gvm_batch":8,"gvm_ensemble":3,"gvm_overlap":3,"refine_threshold":128,"refine_erode":2,"refine_dilate":3,"refine_blur":5,"refine_min_area":500,"refine_temporal_smooth":3,"export_transparent_pngs":true,"export_subject_video":true,"export_matte_video":true,"subject_bg_color":"#000000"}}'
```

## Check Job Status
```bash
curl https://platform.indianetailer.in/v1/inference/{job_id} \
  -H "X-Api-Key: sk-c115b98fe0744fb9b165281a263f58ed"
```
