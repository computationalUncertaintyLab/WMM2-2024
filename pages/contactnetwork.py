import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import networkx as nx
from pyvis.network import Network

def show_contact_network():
    st.title('Contact Network')
    st.markdown('Visualize how people have infected each other within Lehigh University.')

    # Function to generate a random identifier following the specified regex patterns
    def generate_random_id():
        if random.choice([True, False]):
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=3)) + ''.join(random.choices('0123456789', k=3))
        else:
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=2)) + ''.join(random.choices('0123456789', k=2))

    # Generate a fake dataset with some linked nodes
    def generate_fake_dataset(num_entries, overlap_ratio=0.3):
        unique_ids = [generate_random_id() for _ in range(int(num_entries * (1 - overlap_ratio)))]
        overlap_ids = [generate_random_id() for _ in range(int(num_entries * overlap_ratio))]

        data = []
        # Ensure overlap: some nodes are infectors in some cases and infectees in others
        for i in range(len(overlap_ids)):
            infector = overlap_ids[i]
            infectee = random.choice(unique_ids + overlap_ids)
            if infector != infectee:
                data.append((infector, infectee, generate_random_timestamp()))

        # Add remaining isolated nodes
        for _ in range(num_entries - len(data)):
            infector = generate_random_id()
            infectee = generate_random_id()
            if infector != infectee:
                data.append((infector, infectee, generate_random_timestamp()))

        return pd.DataFrame(data, columns=['Infector', 'Infectee', 'Timestamp'])

    # Function to generate a random timestamp within the last year
    def generate_random_timestamp():
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        random_date = start_date + (end_date - start_date) * random.random()
        return random_date.strftime('%Y-%m-%d %H:%M:%S')

    # Initialize session state if not already done
    if 'dataset' not in st.session_state:
        st.session_state.dataset = generate_fake_dataset(30)

    # Retrieve dataset
    df = st.session_state.dataset

    # Combine the generated dataset with any submitted data
    if 'submitted_data' in st.session_state:
        submitted_data = st.session_state.submitted_data
        if not submitted_data.empty and not submitted_data.isna().all().all():
            df = pd.concat([df, submitted_data], ignore_index=True)
    else:
        st.session_state.submitted_data = pd.DataFrame(columns=['Infector', 'Infectee', 'Timestamp'])

    # Debug: Print the combined dataset
    # st.write("Combined dataset after adding submitted data:")
    # st.dataframe(df)

    # Create a directed graph from the dataset
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Infector'], row['Infectee'], timestamp=row['Timestamp'])

    # Debug: Print graph nodes and edges
    # st.write("Graph nodes:")
    # st.write(list(G.nodes))
    # st.write("Graph edges:")
    # st.write(list(G.edges))

    # Create a PyVis network
    def create_pyvis_network(graph, node_colors=None):
        # Set background to white and default node color to black
        net = Network(height='750px', width='100%', bgcolor='white', font_color='black', directed=True)
        
        for node in graph.nodes:
            # If node_colors is provided, use those; otherwise, default to black
            color = node_colors.get(node, 'black') if node_colors else 'black'
            net.add_node(node, label=node, color=color)
        
        for edge in graph.edges:
            net.add_edge(edge[0], edge[1], width=2)
        
        return net


    # Generate and visualize the PyVis network
    net = create_pyvis_network(G)
    net.save_graph('network.html')

    # Display the PyVis network in Streamlit
    def display_pyvis_network(file_path):
        HtmlFile = open(file_path, 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        st.components.v1.html(source_code, height=750)

    display_pyvis_network('network.html')

    # Search for a user and display their contact network
    st.markdown("### Search for a User")
    search_user = st.text_input("Enter a user ID to see their infection details")

    # Create subgraph for the searched user
    def create_subgraph(G, search_user):
        if search_user in G.nodes:
            # Find primary contacts (nodes infected by the searched user)
            primary_contacts = [node for node in G.successors(search_user)]
            # Find secondary contacts (nodes infected by primary contacts)
            secondary_contacts = [neighbor for p in primary_contacts for neighbor in G.successors(p) if neighbor != search_user]

            subgraph_nodes = set([search_user] + primary_contacts + secondary_contacts)
            subgraph = G.subgraph(subgraph_nodes)
            return subgraph, primary_contacts
        else:
            return None, []

    # Display search results and subgraph
    if search_user:
        subgraph, primary_contacts = create_subgraph(G, search_user)
        if subgraph:
            infected_count = df[df['Infector'] == search_user].shape[0]
            infection_dates = df[df['Infector'] == search_user]['Timestamp']
            first_infection = infection_dates.min() if not infection_dates.empty else "No infections"
            
            st.markdown(f"**User: {search_user}**")
            st.markdown(f"- Number of people infected: **{infected_count}**")
            st.markdown(f"- First infection date: **{first_infection}**")
            
            # Assign colors to nodes
            node_colors = {search_user: 'blue'}
            for node in primary_contacts:
                node_colors[node] = 'red'
            for node in subgraph.nodes:
                if node not in node_colors:
                    node_colors[node] = 'gray'
            
            # Visualize subgraph
            subgraph_net = create_pyvis_network(subgraph, node_colors)
            subgraph_net.save_graph('subgraph.html')
            
            display_pyvis_network('subgraph.html')

            st.markdown("**Color Coding in the Subgraph:**")
            st.markdown("- **Blue**: The searched user")
            st.markdown("- **Red**: Users directly infected by the searched user (primary contacts)")
            st.markdown("- **Gray**: Users infected by the primary contacts (secondary contacts)")
        else:
            st.markdown(f"User '{search_user}' not found in the network.")

    # Show the generated data
    st.markdown("### Generated Contact Data")
    st.dataframe(df)
