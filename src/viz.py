import json
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib import gridspec
import numpy as np
from matplotlib.patches import Rectangle
import io
import base64
from pathlib import Path
from typing import Optional
from pydantic import TypeAdapter

from src.models import GRID, COLOR_MAP  # GRID is defined as: list[list[int]]


def viz_grid(
    grid: list[list[int]], color_map: dict[int, str], ax: plt.Axes = None
) -> plt.Axes:
    """
    Visualizes a grid of integer cells as colored squares on a given matplotlib Axes.

    Each integer in the grid is mapped to a color defined in color_map.
    A slight grey border is drawn between cells.

    Parameters:
        grid (list[list[int]]): A 2D list representing the grid of integers.
        color_map (dict[int, str]): A mapping from integer values to color strings.
                                    Example: {0: 'white', 1: 'blue', 2: 'red'}
        ax (plt.Axes, optional): An Axes object to plot on. If None, a new figure and axis are created.

    Returns:
        plt.Axes: The Axes with the plotted grid.
    """
    # Make a local copy of the grid and convert to a NumPy array.
    grid = grid.copy()
    grid_np = np.array(grid)
    rows, cols = grid_np.shape

    # Establish an ordering for the colormap based on sorted keys.
    ordered_keys = sorted(color_map.keys())
    mapping = {val: idx for idx, val in enumerate(ordered_keys)}

    # Map grid values to indices.
    mapped_grid = np.vectorize(mapping.get)(grid_np)

    # Create a ListedColormap and a BoundaryNorm for crisp cell boundaries.
    cmap = mcolors.ListedColormap([color_map[val] for val in ordered_keys])
    norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(ordered_keys) + 0.5, 1), len(ordered_keys)
    )

    # Create an axis if not provided.
    if ax is None:
        fig, ax = plt.subplots()

    # Display the grid.
    ax.imshow(mapped_grid, cmap=cmap, norm=norm)

    # Set tick positions to align gridlines with cell boundaries.
    ax.set_xticks([x - 0.5 for x in range(1 + cols)])
    ax.set_yticks([y - 0.5 for y in range(1 + rows)])

    # Draw gridlines with a slight grey border between cells.
    ax.grid(which="both", color="white", linewidth=3)

    # Remove tick labels.
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Hide the axes spines for a cleaner look.
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax


def viz_many(
    grids: list[list[GRID]], row_border_colors: list[str], color_map: dict[int, str]
) -> None:
    """
    Visualizes multiple grids arranged in rows as specified by a 2D list.

    Each inner list in 'grids' represents one row of subplots.
    For example, given:
        grids = [[G1, G2], [G3, G4], [G5]]
    The function creates three rows of subplots:
      - Row 1: G1 and G2 side by side.
      - Row 2: G3 and G4 side by side.
      - Row 3: G5 is placed in the first column and the remaining subplot(s) are hidden.

    Additionally, each row is outlined with a border whose color is taken from
    row_border_colors (a list of hex color strings, one per row).

    Parameters:
        grids (list[list[list[int]]]): A 2D list where each element is a grid (a 2D list of integers).
        row_border_colors (list[str]): A list of hex color strings. Its length must equal the number of rows.
        color_map (dict[int, str]): A mapping from integer values to color strings.
    """
    n_rows = len(grids)
    n_cols = max(len(row) for row in grids) if grids else 0

    if len(row_border_colors) != n_rows:
        raise ValueError(
            f"Expected {n_rows} row border colors, but got {len(row_border_colors)}."
        )

    # Create a grid of subplots.
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), constrained_layout=True
    )

    # Normalize axs into a 2D array even if there's only one row or column.
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1:
        axs = np.array([axs])
    elif n_cols == 1:
        axs = np.array([[ax] for ax in axs])

    # Plot each grid in its corresponding subplot.
    for i, row in enumerate(grids):
        for j in range(n_cols):
            if j < len(row):
                viz_grid(row[j], color_map, ax=axs[i, j])
            else:
                # Hide any unused subplots.
                axs[i, j].axis("off")

    # Force the layout to update so that we get the correct positions.
    fig.canvas.draw()

    # For each row, compute the union of the axes positions and draw a colored border.
    for i in range(n_rows):
        # Get positions (in figure coordinates) for all axes in the i-th row.
        row_positions = [axs[i, j].get_position() for j in range(n_cols)]
        left = min(pos.x0 for pos in row_positions)
        right = max(pos.x1 for pos in row_positions)
        bottom = min(pos.y0 for pos in row_positions)
        top = max(pos.y1 for pos in row_positions)
        width = right - left
        height = top - bottom

        # Create a rectangle patch with no fill and the specified border color.
        rect = Rectangle(
            (left, bottom),
            width,
            height,
            fill=False,
            edgecolor=row_border_colors[i],
            lw=5,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.add_artist(rect)

    plt.show()


def base64_from_grid(grid: GRID) -> str:
    """
    Converts a grid to a base64-encoded PNG image.

    Parameters:
        grid (GRID): A 2D list representing the grid of integers.

    Returns:
        str: Base64-encoded string of the grid visualization as a PNG image.
    """
    # Create a figure and axis with no padding
    fig, ax = plt.subplots(figsize=(6, 6))

    # Remove all margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Visualize the grid using the existing viz_grid function
    viz_grid(grid, COLOR_MAP, ax=ax)

    # Save the figure to a BytesIO buffer with no padding
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", pad_inches=0, dpi=150)
    buffer.seek(0)

    # Convert to base64
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    # Close the figure to free memory
    plt.close(fig)

    return image_base64


def viz_attempts_interactive(
    attempts_path: Path,
    challenges_path: Path,
    solutions_path: Optional[Path] = None,
    year: str = "2025",
    split: str = "evaluation",
) -> None:
    """
    Interactive visualization of attempts with navigation buttons.
    
    Displays:
    - Training examples (sample puzzles with input/output pairs) at the top
    - Test input puzzle
    - Best attempt (attempt_1 or attempt_2, chosen based on which matches solution better)
    - Actual solution (if available)
    
    Navigation:
    - Use arrow buttons (← Previous / Next →) to navigate between puzzles
    - Keyboard shortcuts: Left/Right arrow keys or A/D keys
    
    Parameters:
        attempts_path: Path to the attempts JSON file
        challenges_path: Path to the challenges JSON file
        solutions_path: Optional path to solutions JSON file. If provided, will show solutions and choose best attempt.
        year: Year for the ARC prize (default: "2025")
        split: Split name (e.g., "evaluation", "training") - default: "evaluation"
    """
    from src.models import Challenge
    from src.run import ChallengeSolution
    
    # Load challenges
    raw_challenges = json.loads(challenges_path.read_text())
    challenges_dict = {
        k: Challenge.model_validate({**v, "task_id": k})
        for k, v in raw_challenges.items()
    }
    
    # Load attempts
    attempts_data = json.loads(attempts_path.read_text())
    attempts_dict = TypeAdapter(dict[str, list[ChallengeSolution]]).validate_json(
        json.dumps(attempts_data)
    )
    
    # Load solutions if provided
    solutions_dict = None
    if solutions_path and solutions_path.exists():
        solutions_data = json.loads(solutions_path.read_text())
        solutions_dict = TypeAdapter(dict[str, list[list[list[int]]]]).validate_json(
            json.dumps(solutions_data)
        )
    
    # Create a list of all puzzle items (one per task_id, using first test input)
    puzzle_items = []
    for task_id in sorted(attempts_dict.keys()):
        if task_id in challenges_dict:
            challenge = challenges_dict[task_id]
            # Only add each task once, using the first test input (test_idx=0)
            if len(challenge.test) > 0 and len(attempts_dict[task_id]) > 0:
                puzzle_items.append((task_id, 0))
    
    if not puzzle_items:
        print("No puzzles found to visualize!")
        return
    
    # Find best attempt by comparing with solution if available
    def get_best_attempt(task_id: str, test_idx: int) -> GRID:
        challenge_solutions = attempts_dict[task_id]
        if test_idx >= len(challenge_solutions):
            return challenge_solutions[0].attempt_1
        
        solution = challenge_solutions[test_idx]
        attempt_1 = solution.attempt_1
        attempt_2 = solution.attempt_2
        
        # If we have solutions, choose the one that matches better
        if solutions_dict and task_id in solutions_dict:
            task_solutions = solutions_dict[task_id]
            if test_idx < len(task_solutions):
                actual_solution = task_solutions[test_idx]
                # Count matching cells
                def count_matches(grid1, grid2):
                    if len(grid1) != len(grid2):
                        return 0
                    matches = 0
                    for i in range(min(len(grid1), len(grid2))):
                        if len(grid1[i]) != len(grid2[i]):
                            return 0
                        for j in range(len(grid1[i])):
                            if grid1[i][j] == grid2[i][j]:
                                matches += 1
                    return matches
                
                matches_1 = count_matches(attempt_1, actual_solution)
                matches_2 = count_matches(attempt_2, actual_solution)
                return attempt_1 if matches_1 >= matches_2 else attempt_2
        
        # Default to attempt_1 if no solution available
        return attempt_1
    
    # Current puzzle index
    current_idx = [0]
    
    # Create figure with subplots
    # Layout: left side (training examples), right side (test input/attempt/solution stacked)
    fig = plt.figure(figsize=(16, 12))
    
    # Grid layout: left side for training examples, right side for test panels
    max_train_examples = max(
        len(challenges_dict[task_id].train) 
        for task_id, _ in puzzle_items
    ) if puzzle_items else 1
    
    # Create grid: 
    # - 3 columns: 2 for left (training input/output), 1 for right (test panels)
    # - Rows: max(max_train_examples, 3) for content + 1 for buttons
    # - Width ratios: left side 40% (2+2), right side 60% (3)
    # - Height ratios: equal for content rows, smaller for button row
    n_rows = max(max_train_examples, 3) + 1  # training examples or min 3, plus button row
    gs = gridspec.GridSpec(
        n_rows, 
        3,  # 3 columns: 2 for training (input/output), 1 for test panels
        figure=fig, 
        width_ratios=[2, 2, 3],  # Left side 40% (2+2=4 out of 7), right side 60% (3 out of 7)
        height_ratios=[1] * max(max_train_examples, 3) + [0.5],  # Equal height for content, smaller for buttons
        hspace=0.4, 
        wspace=0.4
    )
    
    # Create all axes upfront
    # Left side: training examples (input/output pairs)
    train_axes = []
    for i in range(max_train_examples):
        ax_in = fig.add_subplot(gs[i, 0])
        ax_out = fig.add_subplot(gs[i, 1])
        train_axes.append([ax_in, ax_out])
    
    # Right side: test input, attempt, solution (stacked vertically)
    # Always use first 3 rows on the right, regardless of training example count
    ax_input = fig.add_subplot(gs[0, 2])
    ax_attempt = fig.add_subplot(gs[1, 2])
    ax_solution = fig.add_subplot(gs[2, 2])
    
    def update_display():
        """Update the display with current puzzle."""
        task_id, test_idx = puzzle_items[current_idx[0]]
        challenge = challenges_dict[task_id]
        test_input = challenge.test[test_idx].input
        train_examples = challenge.train
        
        # Update training examples (left side)
        for i in range(max_train_examples):
            ax_in, ax_out = train_axes[i]
            ax_in.clear()
            ax_out.clear()
            if i < len(train_examples):
                example = train_examples[i]
                viz_grid(example.input, COLOR_MAP, ax=ax_in)
                ax_in.set_title(f"Example {i+1} Input", fontsize=12, fontweight="bold")
                viz_grid(example.output, COLOR_MAP, ax=ax_out)
                ax_out.set_title(f"Example {i+1} Output", fontsize=12, fontweight="bold")
                ax_in.set_visible(True)
                ax_out.set_visible(True)
            else:
                ax_in.set_visible(False)
                ax_out.set_visible(False)
        
        # Clear and update test axes (right side, stacked vertically)
        ax_input.clear()
        ax_attempt.clear()
        ax_solution.clear()
        
        # Display test input (top right)
        viz_grid(test_input, COLOR_MAP, ax=ax_input)
        ax_input.set_title("Test Input", fontsize=14, fontweight="bold")
        
        # Display best attempt (middle right)
        best_attempt = get_best_attempt(task_id, test_idx)
        viz_grid(best_attempt, COLOR_MAP, ax=ax_attempt)
        
        # Display solution (bottom right)
        if solutions_dict and task_id in solutions_dict:
            task_solutions = solutions_dict[task_id]
            if test_idx < len(task_solutions):
                actual_solution = task_solutions[test_idx]
                viz_grid(actual_solution, COLOR_MAP, ax=ax_solution)
                ax_solution.set_title("Solution", fontsize=14, fontweight="bold")
                
                # Check if attempt matches solution
                if best_attempt == actual_solution:
                    ax_attempt.set_title("Best Attempt ✓", fontsize=14, fontweight="bold", color="green")
                else:
                    ax_attempt.set_title("Best Attempt ✗", fontsize=14, fontweight="bold", color="red")
            else:
                ax_solution.text(0.5, 0.5, "Solution\nNot Available", 
                                ha="center", va="center", fontsize=12)
                ax_solution.set_title("Solution", fontsize=14, fontweight="bold")
                ax_attempt.set_title("Best Attempt", fontsize=14, fontweight="bold")
        else:
            ax_solution.text(0.5, 0.5, "Solution\nNot Available", 
                            ha="center", va="center", fontsize=12)
            ax_solution.set_title("Solution", fontsize=14, fontweight="bold")
            ax_attempt.set_title("Best Attempt", fontsize=14, fontweight="bold")
        
        # Update figure title
        challenge = challenges_dict[task_id]
        num_tests = len(challenge.test)
        test_info = f"Test #{test_idx + 1}" if num_tests > 1 else "Test"
        fig.suptitle(
            f"Puzzle {current_idx[0] + 1}/{len(puzzle_items)} - Task ID: {task_id} - {test_info}",
            fontsize=16,
            fontweight="bold"
        )
        
        fig.canvas.draw()
    
    def next_puzzle(event):
        """Move to next puzzle."""
        if current_idx[0] < len(puzzle_items) - 1:
            current_idx[0] += 1
            update_display()
    
    def prev_puzzle(event):
        """Move to previous puzzle."""
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_display()
    
    # Create navigation buttons (on the button row, bottom)
    button_row_idx = n_rows - 1
    ax_prev = fig.add_subplot(gs[button_row_idx, 0])
    ax_next = fig.add_subplot(gs[button_row_idx, 2])
    ax_prev.axis('off')
    ax_next.axis('off')
    
    btn_prev = Button(ax_prev, "← Previous", color="lightblue", hovercolor="lightcyan")
    btn_next = Button(ax_next, "Next →", color="lightblue", hovercolor="lightcyan")
    
    btn_prev.on_clicked(prev_puzzle)
    btn_next.on_clicked(next_puzzle)
    
    # Add keyboard shortcuts
    def on_key(event):
        """Handle keyboard events."""
        if event.key == 'left' or event.key == 'a':
            prev_puzzle(None)
        elif event.key == 'right' or event.key == 'd':
            next_puzzle(None)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display
    update_display()
    
    # Show the plot
    plt.show()


def viz_attempts_interactive_from_folder(
    year: str = "2025",
    split: str = "evaluation",
    root_dir: Optional[Path] = None,
) -> None:
    """
    Convenience function to visualize attempts using standard folder structure.
    
    Parameters:
        year: Year for the ARC prize (default: "2025")
        split: Split name (e.g., "evaluation", "training") - default: "evaluation"
        root_dir: Root directory of the project. If None, will use parent of src/.
    """
    if root_dir is None:
        root_dir = Path(__file__).parent.parent
    
    attempts_path = root_dir / "attempts" / f"arc-prize-{year}" / f"arc-agi_{split}_attempts.json"
    challenges_path = root_dir / "data" / f"arc-prize-{year}" / f"arc-agi_{split}_challenges.json"
    solutions_path = root_dir / "data" / f"arc-prize-{year}" / f"arc-agi_{split}_solutions.json"
    
    if not attempts_path.exists():
        raise FileNotFoundError(f"Attempts file not found: {attempts_path}")
    if not challenges_path.exists():
        raise FileNotFoundError(f"Challenges file not found: {challenges_path}")
    
    viz_attempts_interactive(
        attempts_path=attempts_path,
        challenges_path=challenges_path,
        solutions_path=solutions_path if solutions_path.exists() else None,
        year=year,
        split=split,
    )


# Example usage:
if __name__ == "__main__":
    # Example 1: Simple visualization
    from src.models import (
        COLOR_MAP,  # e.g., COLOR_MAP = {0: 'white', 1: 'blue', 2: 'red'}
    )

    # Example 2: Interactive visualization of attempts
    # Uncomment to use:
    # viz_attempts_interactive_from_folder(year="2025", split="evaluation")
    viz_attempts_interactive_from_folder(year="2025", split="evaluation")
