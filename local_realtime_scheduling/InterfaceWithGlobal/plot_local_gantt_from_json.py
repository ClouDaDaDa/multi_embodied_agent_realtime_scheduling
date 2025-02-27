import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import TABLEAU_COLORS


def plot_local_gantt(local_schedule):
    # Define colors for operations using TABLEAU_COLORS
    colors = list(TABLEAU_COLORS.values())
    job_color_map = {}

    # Parse the schedule data
    resource_operations = local_schedule["Resource_Operations"]
    time_window = local_schedule["Time_Window"]

    # Collect all end times to determine x-axis range
    max_end_time = time_window[1]
    for resource, operations in resource_operations.items():
        for operation in operations:
            max_end_time = max(max_end_time, operation["Estimated_End_Time"])

    # Extract resource order for Machines and Transbots
    machines = sorted([res for res in resource_operations if res.startswith("Machine")],
                      key=lambda x: int(x.split("_")[1]))
    transbots = sorted([res for res in resource_operations if res.startswith("Transbot")],
                       key=lambda x: int(x.split("_")[1]))
    resources = machines + transbots  # Combined resource order

    # Create the Gantt chart
    fig, ax = plt.subplots(figsize=(12, 8))

    yticks = []
    yticklabels = []

    for idx, resource in enumerate(resources):
        yticks.append(idx)
        yticklabels.append(resource)

        if resource in resource_operations:
            for operation in resource_operations[resource]:
                operation_id = operation["Operation_ID"]
                job_id = operation_id.split("_")[1]  # Extract the job ID

                # Assign color based on job ID
                if job_id not in job_color_map:
                    job_color_map[job_id] = colors[len(job_color_map) % len(colors)]

                color = job_color_map[job_id]
                start_time = operation["Estimated_Start_Time"]
                duration = operation["Estimated_Duration"]
                end_time = operation["Estimated_End_Time"]

                # Plot a rectangle for the operation
                ax.barh(idx, duration, left=start_time, color=color, edgecolor='black', align='center')

    # Configure axes
    ax.set_xlim(time_window[0], max_end_time)
    ax.set_ylim(-0.5, len(resources) - 0.5)
    ax.set_xlabel("Time")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title("Local Schedule Gantt Chart")

    # Add a red dashed line at the end of the time window
    ax.axvline(x=time_window[1], color='red', linestyle='--', linewidth=2, label="Time Window End")

    # Create a legend for the jobs
    legend_patches = [Patch(color=color, label=f"Job {job_id}") for job_id, color in job_color_map.items()]
    ax.legend(handles=legend_patches, title="Jobs", loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    with open("local_schedules/organized_schedule_window_0.json", "r") as f:
        local_schedule = json.load(f)
    plot_local_gantt(local_schedule)
