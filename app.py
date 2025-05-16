import streamlit as st

current_page=st.navigation([st.Page("Home.py"),
                            st.Page("RegPredict.py", title="Revenue Forecast"),
                            st.Page("ClassPredict.py",title="Purchase Category Predictor"),
                            st.Page("Clusters.py",title="Customer Groupings")
                            ])

current_page.run()