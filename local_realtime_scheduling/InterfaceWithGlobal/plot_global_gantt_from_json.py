import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors


def plot_global_gantt(global_schedule_filepath):
    """
    Generate a Gantt chart from the global_schedule.json file.

    Parameters:
        global_schedule_filepath (str): Path to the global_schedule.json file.
    """
    # Load global schedule data from JSON file
    with open(global_schedule_filepath, 'r') as file:
        data = json.load(file)

    # Generate unique colors for each job
    job_colors = list(mcolors.TABLEAU_COLORS.values())

    # Initialize figure and axes
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define function to add operation rectangles
    def add_operation_bar(ax, resource, start, duration, color):
        rect = patches.Rectangle((start, resource), duration, 0.4, color=color, edgecolor="black")
        ax.add_patch(rect)

    # Extract and sort Machines and Transbots by numeric order
    machines = sorted(
        {op["Machine_Assigned"] for job in data["Global_Schedule"]["Jobs"] for op in job["Operations"] if op["Type"] == "Processing"}
    )
    transbots = sorted(
        {op["Robot_Assigned"] for job in data["Global_Schedule"]["Jobs"] for op in job["Operations"] if op["Type"] == "Transport"}
    )

    # Combine sorted resources
    resource_list = machines + transbots
    resources = {resource_name: idx for idx, resource_name in enumerate(resource_list)}

    # Add operations to the Gantt chart
    for job_idx, job in enumerate(data["Global_Schedule"]["Jobs"]):
        color = job_colors[job_idx % len(job_colors)]  # Assign color to each job
        for operation in job["Operations"]:
            if operation["Type"] == "Processing":
                # Add processing operation
                start = operation.get("Estimated_Start_Time", 0)
                duration = operation.get("Estimated_Duration", 0)
                add_operation_bar(ax, resources[operation["Machine_Assigned"]], start, duration, color)
            elif operation["Type"] == "Transport":
                # Add unload transport operation
                if "Unload_Transport" in operation:
                    start = operation["Unload_Transport"].get("Estimated_Start_Time", 0)
                    duration = operation["Unload_Transport"].get("Estimated_Duration", 0)
                    add_operation_bar(ax, resources[operation["Robot_Assigned"]], start, duration, color)
                # Add loaded transport operation
                if "Loaded_Transport" in operation:
                    start = operation["Loaded_Transport"].get("Estimated_Start_Time", 0)
                    duration = operation["Loaded_Transport"].get("Estimated_Duration", 0)
                    add_operation_bar(ax, resources[operation["Robot_Assigned"]], start, duration, color)

    # Set labels for the vertical axis
    plt.yticks(range(len(resource_list)), resource_list)

    # Add gridlines and labels
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_title('Global Schedule Gantt Chart', fontsize=16)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Resources', fontsize=14)

    # Calculate max end time
    max_end_time = max(
        op.get("Estimated_End_Time", op.get("Estimated_Start_Time", 0) + op.get("Estimated_Duration", 0))
        for job in data["Global_Schedule"]["Jobs"] for op in job["Operations"]
    )
    ax.set_xlim(0, max_end_time)
    ax.set_xticks(range(0, int(max_end_time) + 100, 100))
    ax.set_ylim(-1, len(resource_list))

    # Add legend for job colors
    legend_patches = [patches.Patch(color=job_colors[i % len(job_colors)], label=f"Job {i}") for i in range(len(data["Global_Schedule"]["Jobs"]))]
    ax.legend(handles=legend_patches, loc='upper right', title="Jobs")

    # Display the plot
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == '__main__':
    # Provide the path to the global_schedule.json file
    global_schedule_filepath = 'global_schedule.json'
    plot_global_gantt(global_schedule_filepath)