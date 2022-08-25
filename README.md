# template-script

readme for your automation script

# TODO: try new-baselines


```python
# Initialize Supervisely
# api = sly.Api.from_env()
# project_id = int(os.environ["modal.state.slyProjectId"])
# src_project_info = api.project.get_info_by_id(project_id)
# workspace_id = src_project_info.workspace_id
# src_project_name = src_project_info.name
# dst_project_name = f"{src_project_name}_labeled_inst_seg"

# # Clone source project without existing annotations
# clone_task_id = api.project.clone_advanced(
#     project_id, workspace_id, dst_project_name, with_annotations=False
# )
# api.task.wait(clone_task_id, api.task.Status("finished"))
# dst_project_info = api.project.get_info_by_name(workspace_id, dst_project_name)

# # Get new project meta
# project_meta_json = api.project.get_meta(dst_project_info.id)
# project_meta = sly.ProjectMeta.from_json(project_meta_json)


# def main():
#     model, class_names = load_model()
#     merged_project_meta = get_model_classes(class_names)

#     datasets = api.dataset.get_list(dst_project_info.id)
#     imgs_num = sum([dataset.images_count for dataset in datasets])
#     with tqdm(total=imgs_num) as pbar:
#         for dataset_info in datasets:
#             ds_images = api.image.get_list(dataset_info.id)
#             for image_info in ds_images:
#                 image = api.image.download_np(image_info.id)
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#                 annotation = apply_model_to_image(
#                     image, model, class_names, merged_project_meta
#                 )
#                 api.annotation.upload_ann(image_info.id, annotation)
#                 pbar.update()

```