import streamlit as st
import pandas as pd
import random
from datetime import datetime, timedelta
import networkx as nx
from pyvis.network import Network
import boto3


def show_contact_network():
    st.title('Contact Network')
    st.markdown('Visualize how people have infected each other within Lehigh University.')

     # Initialize session state if not already done
    if 'dataset' not in st.session_state:
        AWS_S3_BUCKET = "wmm2-2024"
        AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
        AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]

        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        st.session_state.dataset = pd.read_csv(f"s3://{AWS_S3_BUCKET}/wmm_live.csv"
                                               ,storage_options={"key"   : AWS_ACCESS_KEY_ID,"secret": AWS_SECRET_ACCESS_KEY})

    if 'dataset' in st.session_state:
        df = st.session_state.dataset
        
    # Combine the generated dataset with any submitted data
    if 'submitted_data' in st.session_state:
        submitted_data = st.session_state.submitted_data
        if not submitted_data.empty and not submitted_data.isna().all().all():
            df = pd.concat([df, submitted_data], ignore_index=True)
    else:
        st.session_state.submitted_data = pd.DataFrame(columns=['Infector', 'Infectee', 'Timestamp'])

    # Create a directed graph from the dataset
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['Infector'], row['Infectee'], timestamp=row['Timestamp'])

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
    search_user = st.text_input("Enter a username to see their infection details")

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
    st.markdown("### Contact Data")
    st.dataframe(df)

if __name__ == "__main__":

    show_contact_network()

