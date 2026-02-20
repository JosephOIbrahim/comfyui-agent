Get-ChildItem -Path 'G:\COMFYUI_Database' -Recurse -Filter 'main.py' -Depth 3 -ErrorAction SilentlyContinue |
    Where-Object { $_.DirectoryName -like '*ComfyUI*' -or $_.DirectoryName -like '*comfy*' } |
    Select-Object -First 5 -ExpandProperty FullName
