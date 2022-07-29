import os
import supervisely as sly
from dotenv import load_dotenv

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

# test credentials
teams = api.team.get_list()
print(f"i'm a member of {len(teams)} teams")

# test value from local.env
project_id = int(os.environ["modal.state.slyProjectId"])
project = api.project.get_info_by_id(project_id)
if project is None:
    print(f"Project with ID {project_id} not found in your account")
else:
    print(f"Project info: {project.name} (id={project.id})")
