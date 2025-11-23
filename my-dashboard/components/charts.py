import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_distributions(df, selected_numeric):
    st.subheader("Distribution Analysis")
    num_charts = len(selected_numeric)
    if num_charts == 1:
        fig = px.histogram(df, x=selected_numeric[0],
                           title=f"Distribution of {selected_numeric[0]}",
                           marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    else:
        cols_per_row = 2
        for i in range(0, num_charts, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col_name in enumerate(selected_numeric[i:i+cols_per_row]):
                with cols[j]:
                    fig = px.histogram(df, x=col_name, title=f"{col_name}", height=300)
                    st.plotly_chart(fig, use_container_width=True)


def render_boxplots(df, selected_numeric):
    st.subheader("Box Plot Comparison")
    fig = go.Figure()
    for col in selected_numeric:
        fig.add_trace(go.Box(y=df[col], name=col))
    fig.update_layout(title="Box Plot Comparison", showlegend=True, height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_corr(df, selected_numeric):
    st.subheader("Correlation Matrix")
    corr_matrix = df[selected_numeric].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)


def render_scatter(df, selected_numeric):
    st.subheader("Scatter Plot Analysis")
    scatter_col1, scatter_col2 = st.columns(2)
    with scatter_col1:
        x_axis = st.selectbox("X-axis", selected_numeric, index=0)
    with scatter_col2:
        y_axis = st.selectbox("Y-axis", selected_numeric, index=min(1, len(selected_numeric)-1))
    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)


