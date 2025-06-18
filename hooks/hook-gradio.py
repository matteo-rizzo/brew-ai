from PyInstaller.utils.hooks import copy_metadata, collect_data_files

# This hook tells PyInstaller to find all data files associated with
# the 'gradio' and 'gradio_client' packages and include them in the build.
# This is the most robust way to handle complex libraries like Gradio.

datas = []
datas += copy_metadata("gradio")
datas += collect_data_files("gradio")
datas += copy_metadata("gradio_client")
datas += collect_data_files("gradio_client")