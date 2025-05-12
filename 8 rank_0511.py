import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set output path to desktop
desktop_path = os.path.expanduser('~/Desktop/')

# Load CSV file
df = pd.read_csv(os.path.join(desktop_path, 'positive_negative_metrics.csv'))

# Process metrics - mark positive metrics where:
# - Index < 164 AND regression_slope > 0, OR
# - Index >= 164 AND regression_slope < 0
df['is_positive'] = np.where(
    ((df.index < 164) & (df['regression_slope'] > 0)) |
    ((df.index >= 164) & (df['regression_slope'] < 0)),
    True, False
)

# Mark negative metrics (index >= 164 and regression_slope < 0)
df['is_negative_mark'] = np.where(
    (df.index >= 164) & (df['regression_slope'] < 0),
    True, False
)

# Filter positive metrics
positive_metrics = df[df['is_positive']]

# Create mapping table to map categories to correct position in table
category_mapping = {
    'proself-SA': {'row': 3, 'col': 'SA', 'base': 'pro-self'},
    'proself-O': {'row': 3, 'col': 'O', 'base': 'pro-self'},
    'propeers-SA': {'row': 4, 'col': 'SA', 'base': 'pro-peers'},
    'propeers-O': {'row': 4, 'col': 'O', 'base': 'pro-peers'},
    'procustomer-SA': {'row': 5, 'col': 'SA', 'base': 'pro-customers'},
    'procustomer-O': {'row': 5, 'col': 'O', 'base': 'pro-customers'},
    'projob-SA': {'row': 6, 'col': 'SA', 'base': 'pro-job'},
    'projob-O': {'row': 6, 'col': 'K', 'base': 'pro-job'}  # Note: projob-O corresponds to K column in framework
}


# Function to extract and clean English metric names
def clean_metric_name(full_name):
    # Extract English part
    english_part = ""

    # If name contains English
    if re.search(r'[a-zA-Z]', full_name):
        # Try to extract English part until first Chinese character
        english_words = []
        for word in full_name.split():
            if all(ord(c) < 128 for c in word):  # Only ASCII characters
                english_words.append(word)
            else:
                break

        if english_words:
            english_part = ' '.join(english_words)
        else:
            # If above method fails, try direct matching of English characters
            match = re.search(r'([A-Za-z\s\-\(\)]+)', full_name)
            if match:
                english_part = match.group(1).strip()
            else:
                english_part = full_name
    else:
        english_part = full_name

    # 1. Remove parentheses and content, like (CSI)
    english_part = re.sub(r'\([^)]*\)', '', english_part)

    # 2. Remove comma and subsequent content
    english_part = re.sub(r',.*', '', english_part)

    # 3. Remove common suffixes
    suffixes = [
        "Scale", "Questionnaire", "Inventory", "Measure", "Index",
        "Survey", "Test", "Items", "Styles", "Adjustment",
        "Reflexivity", "Empowerment", "Exploration", "Behavior",
        "Growth", "Capital", "Attraction", "Examination",
        "Intelligence", "Delay"
    ]

    for suffix in suffixes:
        english_part = re.sub(r'\b' + suffix + r'\b', '', english_part)

    # Clean extra spaces
    english_part = re.sub(r'\s+', ' ', english_part).strip()

    # Ensure not returning empty string
    return english_part if english_part else full_name


# Create a new DataFrame to store all metric information
all_metrics_info = []

for _, metric in positive_metrics.iterrows():
    category = metric['分类']
    if category in category_mapping:
        mapping = category_mapping[category]

        # Clean metric name
        clean_name = clean_metric_name(metric['metric_name'])

        # Add (-) mark for negative metrics
        if metric['is_negative_mark']:
            clean_name += " (-)"

        # Store metric information
        all_metrics_info.append({
            'name': clean_name,
            'category': category,
            'base_category': mapping['base'],
            'col': mapping['col'],
            'regression_slope': metric['regression_slope'],
            'is_negative_mark': metric['is_negative_mark']
        })

# Create DataFrame
metrics_df = pd.DataFrame(all_metrics_info)

# Create 8 section visualizations, each for K, SA or O columns of pro-self, pro-peers, pro-customers, pro-job
sections = [
    {'base': 'pro-self', 'col': 'SA', 'title': 'Pro-self SA Metrics'},
    {'base': 'pro-self', 'col': 'O', 'title': 'Pro-self O Metrics'},
    {'base': 'pro-peers', 'col': 'SA', 'title': 'Pro-peers SA Metrics'},
    {'base': 'pro-peers', 'col': 'O', 'title': 'Pro-peers O Metrics'},
    {'base': 'pro-customers', 'col': 'SA', 'title': 'Pro-customers SA Metrics'},
    {'base': 'pro-customers', 'col': 'O', 'title': 'Pro-customers O Metrics'},
    {'base': 'pro-job', 'col': 'SA', 'title': 'Pro-job SA Metrics'},
    {'base': 'pro-job', 'col': 'K', 'title': 'Pro-job K Metrics'},
]

# Set color maps, different categories use different colors
color_mapping = {
    'pro-self': '#6a98d0',  # Blues
    'pro-peers': '#8bc34a',  # Greens
    'pro-customers': '#ff9800',  # Oranges
    'pro-job': '#f44336'  # Reds
}

# Create visualization for each section
for section in sections:
    # Filter metrics for current section
    mask = (metrics_df['base_category'] == section['base']) & (metrics_df['col'] == section['col'])
    section_df = metrics_df[mask].copy()

    # Sort data
    section_df = section_df.sort_values(by='regression_slope', ascending=False)

    # Skip if no data
    if section_df.empty:
        print(f"No data for {section['title']}")
        continue

    # Create visualization with better proportions
    plt.figure(figsize=(14, max(6, len(section_df) * 0.4)))

    # Get color from mapping
    bar_color = color_mapping[section['base']]

    # Create barplot with better formatting
    ax = sns.barplot(
        x='regression_slope',
        y='name',
        data=section_df,
        color=bar_color
    )

    # Calculate the max absolute value for setting margins
    max_abs_value = max(abs(section_df['regression_slope'].max()), abs(section_df['regression_slope'].min()))

    # Set margins for x-axis - make sure we have enough space for labels
    margin = max_abs_value * 0.25  # 25% margin
    x_min = min(section_df['regression_slope'].min() - margin, 0)
    x_max = max(section_df['regression_slope'].max() + margin, 0)

    # If we have both positive and negative values, ensure 0 is included
    if section_df['regression_slope'].min() < 0 and section_df['regression_slope'].max() > 0:
        plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

    # Set the axis limits
    plt.xlim(x_min, x_max)

    # Add value labels directly on the bars with correct alignment
    for i, p in enumerate(ax.patches):
        value = section_df['regression_slope'].iloc[i]

        # Calculate the label position
        if value >= 0:
            # For positive values, place at the end of the bar
            x_pos = p.get_width()
            ha = 'left'
            x_offset = max_abs_value * 0.01  # Small offset to right
        else:
            # For negative values, place at the end of the bar
            x_pos = p.get_width()
            ha = 'right'
            x_offset = -max_abs_value * 0.01  # Small offset to left

        # Add the text label with proper positioning
        ax.text(
            x_pos + x_offset,  # Position
            p.get_y() + p.get_height() / 2,  # Center of bar
            f"{value:.4f}",  # Formatted value
            ha=ha,  # Horizontal alignment
            va='center',  # Vertical alignment
            fontsize=10
        )

    # Set title and labels
    plt.title(f"{section['title']} (Sorted by regression slope)", fontsize=16, pad=20)
    plt.xlabel('Regression Slope', fontsize=14)
    plt.ylabel('', fontsize=14)  # Empty y-label as names are displayed on y-axis

    # Grid only on x-axis for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Remove top and right spines for cleaner look
    sns.despine()

    # Adjust layout
    plt.tight_layout()

    # Save chart with higher DPI for better quality
    output_file = os.path.join(desktop_path, f"{section['base']}_{section['col']}_metrics.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Chart for {section['title']} saved to {output_file}")

# Create HTML file to display all charts
html_file = os.path.join(desktop_path, 'workplace_ksaos_charts.html')

html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Workplace KSAOs Metrics Charts</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.5;
            max-width: 1200px;
            margin: 0 auto;
            background-color: #f9f9f9;
            color: #333;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #4a6fa5;
            margin-bottom: 30px;
        }
        h2 {
            color: #2c4c7c;
            margin-top: 40px;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .chart-container {
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            background-color: white;
            padding: 15px;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .pro-self {
            border-left: 4px solid #6a98d0;
            padding-left: 15px;
        }
        .pro-peers {
            border-left: 4px solid #8bc34a;
            padding-left: 15px;
        }
        .pro-customers {
            border-left: 4px solid #ff9800;
            padding-left: 15px;
        }
        .pro-job {
            border-left: 4px solid #f44336;
            padding-left: 15px;
        }
        .print-button {
            background-color: #4a6fa5;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 20px 0;
            cursor: pointer;
            border-radius: 4px;
        }
        @media print {
            .print-button {
                display: none;
            }
            img {
                max-width: 100%;
                page-break-inside: avoid;
            }
            h2 {
                page-break-before: always;
            }
            h2:first-of-type {
                page-break-before: avoid;
            }
            .chart-container {
                box-shadow: none;
                border: 1px solid #eee;
            }
        }
    </style>
</head>
<body>
    <h1>Workplace KSAOs Metrics Charts</h1>
    <button class="print-button" onclick="window.print()">Print/Save as PDF</button>
"""

# Organize charts by category
for category in ['pro-self', 'pro-peers', 'pro-customers', 'pro-job']:
    html_content += f'<h2 class="{category}">{category}</h2>\n'

    # Add charts for current category
    for col in ['SA', 'O', 'K']:
        # For pro-job, K column instead of O column
        if category == 'pro-job' and col == 'O':
            continue
        if category == 'pro-job' and col == 'K':
            title = f"{category} K Metrics"
        else:
            # Skip K column for categories other than pro-job
            if col == 'K' and category != 'pro-job':
                continue
            title = f"{category} {col} Metrics"

        file_name = f"{category}_{col}_metrics.png"
        file_path = os.path.join(desktop_path, file_name)

        # Check if file exists
        if os.path.exists(file_path):
            html_content += f'<div class="chart-container">\n'
            html_content += f'    <img src="{file_name}" alt="{title}">\n'
            html_content += f'</div>\n'

html_content += """
</body>
</html>
"""

# Save HTML file
with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"HTML file successfully created at: {html_file}")
print("This HTML file contains visualizations for all 8 sections and can be viewed in a browser or printed as PDF")