from google.cloud import aiplatform

PROJECT_ID = "profitscout-lx6bb"
REGION = "us-central1"

aiplatform.init(project=PROJECT_ID, location=REGION)

def delete_old_schedules(display_name):
    print(f"Searching for existing schedules with name: {display_name}...")
    schedules = aiplatform.PipelineJobSchedule.list(filter=f'display_name="{display_name}"')
    
    if not schedules:
        print("No existing schedules found.")
        return

    for schedule in schedules:
        print(f"Deleting schedule: {schedule.resource_name} (State: {schedule.state})")
        try:
            schedule.pause()
            schedule.delete()
            print("Deleted.")
        except Exception as e:
            print(f"Error deleting {schedule.resource_name}: {e}")

if __name__ == "__main__":
    delete_old_schedules("profitscout-weekly-training")
