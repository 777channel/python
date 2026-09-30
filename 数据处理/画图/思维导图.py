from graphviz import Digraph


def create_mind_map():
    # Create a new directed graph
    dot = Digraph(comment='Mind Mapping', format='png')

    # Global graph settings
    dot.graph_attr['rankdir'] = 'TB'  # Top to Bottom layout
    dot.node_attr['shape'] = 'box'
    dot.node_attr['style'] = 'filled'
    dot.node_attr['fontname'] = 'Arial'

    # Customizable color palette
    colors = {
        'background': '#4A90E2',  # Main blue
        'node_bg': '#FFFFFF',  # White
        'node_border': '#2C3E50',  # Dark grey
        'text': '#000000',  # Black
    }

    # Create the mind map
    # Root node
    dot.node('root', 'Mind Mapping',
             fillcolor=colors['background'],
             color=colors['node_border'],
             fontcolor=colors['text'],
             fontsize='16')

    # Main branches
    main_branches = [
        ('def', 'Definition'),
        ('prin', 'Core Principles'),
        ('adv', 'Advantages'),
        ('method', 'Learning Method'),
        ('impact', 'Personal Impact')
    ]

    # Sub-branches
    sub_branches = {
        'def': ['Visual Learning Tool', 'Brain-Friendly Technique'],
        'prin': ['Central Topic', 'Radial Branches', 'Keywords Only', 'Personal Insights'],
        'adv': ['Better Memory', 'Quick Understanding', 'Easy Retrieval'],
        'method': ['Handwritten', 'One-Page Format', 'Flexible Structure'],
        'impact': ['Academic Success', 'Business Application', 'Teaching Others']
    }

    # Add main branches
    for key, label in main_branches:
        dot.node(key, label,
                 fillcolor=colors['node_bg'],
                 color=colors['node_border'],
                 fontcolor=colors['text'])
        dot.edge('root', key)

    # Add sub-branches
    for parent_key, subs in sub_branches.items():
        for i, sub in enumerate(subs):
            sub_key = f'{parent_key}_sub_{i}'
            dot.node(sub_key, sub,
                     fillcolor=colors['node_bg'],
                     color=colors['node_border'],
                     fontcolor=colors['text'])
            dot.edge(parent_key, sub_key)

    # Render the graph
    dot.render('mind_map', cleanup=True)
    print("Mind map created successfully!")


# Customize colors and generate mind map
def main():
    create_mind_map()


if __name__ == '__main__':
    main()

# Usage instructions:
# 1. Install graphviz: pip install graphviz
# 2. Modify the 'colors' dictionary to change color scheme
# 3. Run the script to generate mind_map.png