#!/usr/bin/env python
# coding: utf-8

# In[20]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch and parse data from the general election results page
def fetch_general_election_results(url):
    print(f"Fetching data from {url}...")
    
    # Fetch the HTML content of the page
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully accessed the page: {url}")
    else:
        print(f"Failed to access the page: {url}. Status code: {response.status_code}")
        return
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    print("HTML content fetched and parsed.")
    
    # Extract data from the first table in the page
    results_table = soup.find('table', class_='table')
    if results_table is None:
        print("No results table found.")
        return

    print("Results table found.")
    
    # Initialize lists to store extracted data
    parties = []
    won = []
    leading = []
    total = []

    for row in results_table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 4:
            print("Skipping row with insufficient columns.")
            continue  # Skip rows that do not have enough columns

        parties.append(cells[0].text.strip())  # Party name
        won.append(cells[1].text.strip())     # Number of seats won
        leading.append(cells[2].text.strip()) # Number of seats leading
        total.append(cells[3].text.strip())   # Total seats contested

    print(f"Extracted data for {len(parties)} parties.")
    
    # Create a DataFrame to store the extracted data
    df = pd.DataFrame({
        'Party': parties,
        'Seats Won': won,
        'Seats Leading': leading,
        'Total Seats': total
    })

    # Convert columns to appropriate data types
    df['Seats Won'] = df['Seats Won'].astype(int)
    df['Seats Leading'] = df['Seats Leading'].astype(int)
    df['Total Seats'] = df['Total Seats'].astype(int)

    # Save the data to a CSV file
    csv_filename = 'general_election_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}.")
    
    # Display the DataFrame
    print(df.head())  # Print the first few rows of the DataFrame to check the data
    
    # Plot the data
    plot_general_election_results(df)
    
    return df

# Function to plot pie chart and bar chart for the general election results
def plot_general_election_results(df):
    df_filtered = df[['Party', 'Seats Won']].copy()
    df_filtered['Percent'] = (df_filtered['Seats Won'] / df_filtered['Seats Won'].sum()) * 100
    
    # Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(df_filtered['Seats Won'], labels=df_filtered['Party'], autopct='%1.1f%%', startangle=140)
    plt.title('General Election Results Distribution')
    plt.savefig('general_pie_chart.png')
    plt.show()

    # Bar Chart
    plt.figure(figsize=(12, 6))
    plt.bar(df_filtered['Party'], df_filtered['Seats Won'])
    plt.title('Number of Seats Won by Each Party')
    plt.xlabel('Party')
    plt.ylabel('Number of Seats')
    plt.xticks(rotation=90)
    plt.savefig('general_bar_chart.png')
    plt.show()

# Function to fetch and parse data from the bye-election results page
def fetch_by_election_results(url):
    print(f"Fetching data from {url}...")
    
    # Fetch the HTML content of the page
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully accessed the page: {url}")
    else:
        print(f"Failed to access the page: {url}. Status code: {response.status_code}")
        return None  # Return None if page access fails
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    print("HTML content fetched and parsed.")
    
    # Extract data from the div elements with class 'const-box'
    const_boxes = soup.find_all('div', class_='const-box')
    
    if not const_boxes:
        print("No constituency boxes found.")
        return None
    
    print(f"Found {len(const_boxes)} constituency boxes.")
    
    # Initialize lists to store extracted data
    constituencies = []
    states = []
    results = []
    candidates = []
    parties = []

    for box in const_boxes:
        constituency = box.find('h3').text.strip()
        state = box.find('h4').text.strip()
        result = box.find('h2').text.strip()
        candidate = box.find('h5').text.strip()
        party = box.find('h6').text.strip()
        
        constituencies.append(constituency)
        states.append(state)
        results.append(result)
        candidates.append(candidate)
        parties.append(party)

    print(f"Extracted data for {len(constituencies)} constituencies.")
    
    df = pd.DataFrame({
        'Constituency': constituencies,
        'State': states,
        'Result': results,
        'Candidate': candidates,
        'Party': parties
    })

    # Save the data to a CSV file
    csv_filename = url.split('/')[-2] + '_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}.")
    
    # Display the DataFrame
    print(df.head())  # Print the first few rows of the DataFrame to check the data
    
    # Plot the data
    plot_bye_election_results(df)
    
    return df

# Function to plot pie chart and bar chart for the bye-election results
def plot_bye_election_results(df):
    # Pie Chart
    result_counts = df['Result'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Bye-Election Results Distribution')
    plt.savefig('bye_pie_chart.png')
    plt.show()

    # Bar Chart
    party_counts = df['Party'].value_counts()
    plt.figure(figsize=(10, 6))
    party_counts.plot(kind='bar')
    plt.title('Number of Seats Won by Each Party')
    plt.xlabel('Party')
    plt.ylabel('Number of Seats')
    plt.xticks(rotation=90)
    plt.savefig('bye_bar_chart.png')
    plt.show()

# Function to fetch data from multiple pages
def fetch_data_from_multiple_urls(urls, fetch_function):
    for url in urls:
        df = fetch_function(url)
        if df is not None:  # Ensure we only save if df is valid
            # Save data to CSV
            csv_filename = url.split('/')[-2] + '_results.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}.")

# Define the URLs
general_election_url = 'https://results.eci.gov.in/PcResultGenJune2024/index.htm'

# Fetch data from the general election results page
fetch_general_election_results(general_election_url)

# Define the URLs for bye-elections
bye_election_urls = [
    'https://results.eci.gov.in/AcResultByeJune2024/'
 
]

# Fetch data from the bye-election results pages
fetch_data_from_multiple_urls(bye_election_urls, fetch_by_election_results)


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
csv_filename = 'general_election_results.csv'
df = pd.read_csv(csv_filename)

# Display the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Total seats won by each party
print("\nTotal seats won by each party:")
print(df[['Party', 'Seats Won']])

# Plot the number of seats won by each party
def plot_seats_won(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df['Party'], df['Seats Won'], color='skyblue')
    plt.title('Number of Seats Won by Each Party')
    plt.xlabel('Party')
    plt.ylabel('Number of Seats')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('seats_won_by_party.png')
    plt.show()

# Call the plot function
plot_seats_won(df)

# Additional insights
total_seats = df['Seats Won'].sum()
print(f"\nTotal seats contested: {total_seats}")

# Party with the most seats won
max_seats = df['Seats Won'].max()
party_max_seats = df[df['Seats Won'] == max_seats]['Party'].values[0]
print(f"The party with the most seats won is {party_max_seats} with {max_seats} seats.")

# Seats leading by each party
print("\nSeats leading by each party:")
print(df[['Party', 'Seats Leading']])

# Plot the number of seats leading by each party
def plot_seats_leading(df):
    plt.figure(figsize=(10, 6))
    plt.bar(df['Party'], df['Seats Leading'], color='lightcoral')
    plt.title('Number of Seats Leading by Each Party')
    plt.xlabel('Party')
    plt.ylabel('Number of Seats Leading')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('seats_leading_by_party.png')
    plt.show()

# Call the plot function
plot_seats_leading(df)

# Percentage of total seats won by each party
df['Percentage of Total Seats'] = (df['Seats Won'] / total_seats) * 100
print("\nPercentage of total seats won by each party:")
print(df[['Party', 'Percentage of Total Seats']])

# Plot the percentage of total seats won by each party
def plot_percentage_seats_won(df):
    plt.figure(figsize=(10, 6))
    plt.pie(df['Percentage of Total Seats'], labels=df['Party'], autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Total Seats Won by Each Party')
    plt.tight_layout()
    plt.savefig('percentage_seats_won_by_party.png')
    plt.show()

# Call the plot function
plot_percentage_seats_won(df)


# In[25]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch and parse data from the election results page
def fetch_election_results(url):
    print(f"Fetching data from {url}...")
    
    # Fetch the HTML content of the page
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully accessed the page: {url}")
    else:
        print(f"Failed to access the page: {url}. Status code: {response.status_code}")
        return
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    print("HTML content fetched and parsed.")
    
    # Extract data from the table inside the nested div elements
    results_table = soup.find('div', class_='rslt-table').find('table', class_='table')
    if results_table is None:
        print("No results table found.")
        return
    
    print("Results table found.")
    
    # Initialize lists to store extracted data
    parties = []
    won = []
    leading = []
    total = []

    for row in results_table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 4:
            print("Skipping row with insufficient columns.")
            continue  # Skip rows that do not have enough columns
        
        party_name = cells[0].text.strip()  # Party name
        won_seats = cells[1].text.strip()  # Number of seats won
        leading_seats = cells[2].text.strip()  # Number of seats leading
        total_seats = cells[3].text.strip()  # Total seats contested
        
        parties.append(party_name)
        won.append(int(won_seats))  # Convert to integer
        leading.append(int(leading_seats))  # Convert to integer
        total.append(int(total_seats))  # Convert to integer

    print(f"Extracted data for {len(parties)} parties.")
    
    # Create a DataFrame to store the extracted data
    df = pd.DataFrame({
        'Party': parties,
        'Seats Won': won,
        'Seats Leading': leading,
        'Total Seats': total
    })

    # Save the data to a CSV file
    csv_filename = 'party_wise_election_results.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}.")
    
    # Display the DataFrame
    print(df.head())  # Print the first few rows of the DataFrame to check the data
    
    # Plot the data
    plot_election_results(df)
    
    return df

# Function to plot pie chart and bar chart for the election results
def plot_election_results(df):
    df_filtered = df[['Party', 'Seats Won']].copy()
    df_filtered['Percent'] = (df_filtered['Seats Won'] / df_filtered['Seats Won'].sum()) * 100
    
    # Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie(df_filtered['Seats Won'], labels=df_filtered['Party'], autopct='%1.1f%%', startangle=140)
    plt.title('Election Results Distribution')
    plt.savefig('party_wise_pie_chart.png')
    plt.show()

    # Bar Chart
    plt.figure(figsize=(12, 6))
    plt.bar(df_filtered['Party'], df_filtered['Seats Won'])
    plt.title('Number of Seats Won by Each Party')
    plt.xlabel('Party')
    plt.ylabel('Number of Seats')
    plt.xticks(rotation=90)
    plt.savefig('party_wise_bar_chart.png')
    plt.show()

# Fetch data from the election results page
url = 'https://results.eci.gov.in/AcResultGenJune2024/partywiseresult-S01.htm'
df = fetch_election_results(url)

# Analyze the fetched data
if df is not None:
    print("Data analysis:")
    # Display the first few rows of the DataFrame
    print("First few rows of the DataFrame:")
    print(df.head())

    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe())

    # Total seats won by each party
    print("\nTotal seats won by each party:")
    print(df[['Party', 'Seats Won']])

    # Additional insights
    total_seats = df['Seats Won'].sum()
    print(f"\nTotal seats contested: {total_seats}")

    # Party with the most seats won
    max_seats = df['Seats Won'].max()
    party_max_seats = df[df['Seats Won'] == max_seats]['Party'].values[0]
    print(f"The party with the most seats won is {party_max_seats} with {max_seats} seats.")

    # Seats leading by each party
    print("\nSeats leading by each party:")
    print(df[['Party', 'Seats Leading']])

    # Percentage of total seats won by each party
    df['Percentage of Total Seats'] = (df['Seats Won'] / total_seats) * 100
    print("\nPercentage of total seats won by each party:")
    print(df[['Party', 'Percentage of Total Seats']])


# In[26]:


import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('party_wise_election_results.csv')
print(df.head())


# In[27]:


# Summary statistics
print("Summary statistics:")
print(df.describe())

# Total seats won
total_seats_won = df['Seats Won'].sum()
print(f"\nTotal seats won by all parties: {total_seats_won}")

# Party with the maximum seats won
max_seats = df['Seats Won'].max()
party_max_seats = df[df['Seats Won'] == max_seats]['Party'].values[0]
print(f"The party with the most seats won is {party_max_seats} with {max_seats} seats.")

# Party with the least seats won
min_seats = df['Seats Won'].min()
party_min_seats = df[df['Seats Won'] == min_seats]['Party'].values[0]
print(f"The party with the least seats won is {party_min_seats} with {min_seats} seats.")


# In[28]:


# Add percentage of total seats
df['Percentage of Total Seats'] = (df['Seats Won'] / total_seats_won) * 100

print("\nPercentage of total seats won by each party:")
print(df[['Party', 'Percentage of Total Seats']])

# Highest percentage of seats won
max_percentage = df['Percentage of Total Seats'].max()
party_max_percentage = df[df['Percentage of Total Seats'] == max_percentage]['Party'].values[0]
print(f"The party with the highest percentage of total seats won is {party_max_percentage} with {max_percentage:.2f}% of the seats.")


# In[30]:


import matplotlib.pyplot as plt

# Pie Chart for Seats Won
plt.figure(figsize=(10, 8))
plt.pie(df['Seats Won'], labels=df['Party'], autopct='%1.1f%%', startangle=140)
plt.title('Election Results Distribution')
plt.savefig('party_wise_pie_chart.png')
plt.show()

# Bar Chart for Number of Seats Won
plt.figure(figsize=(14, 7))
plt.bar(df['Party'], df['Seats Won'], color='skyblue')
plt.title('Number of Seats Won by Each Party')
plt.xlabel('Party')
plt.ylabel('Number of Seats')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('party_wise_bar_chart.png')
plt.show()


# In[35]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch and parse data from the party-wise results page
def fetch_party_wise_results(url):
    print(f"Fetching data from {url}...")
    
    # Fetch the HTML content of the page
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully accessed the page: {url}")
    else:
        print(f"Failed to access the page: {url}. Status code: {response.status_code}")
        return
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    print("HTML content fetched and parsed.")
    
    # Extract data from the first table in the page
    results_table = soup.find('table', class_='table')
    if results_table is None:
        print("No results table found.")
        return

    print("Results table found.")
    
    # Initialize lists to store extracted data
    parties = []
    won = []
    leading = []
    total = []

    for row in results_table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 4:
            print("Skipping row with insufficient columns.")
            continue  # Skip rows that do not have enough columns

        parties.append(cells[0].text.strip())  # Party name
        won.append(cells[1].text.strip())     # Number of seats won
        leading.append(cells[2].text.strip()) # Number of seats leading
        total.append(cells[3].text.strip())   # Total seats contested

    print(f"Extracted data for {len(parties)} parties.")
    
    # Create a DataFrame to store the extracted data
    df = pd.DataFrame({
        'Party': parties,
        'Seats Won': won,
        'Seats Leading': leading,
        'Total Seats': total
    })

    # Convert columns to appropriate data types
    df['Seats Won'] = df['Seats Won'].astype(int)
    df['Seats Leading'] = df['Seats Leading'].astype(int)
    df['Total Seats'] = df['Total Seats'].astype(int)

    # Save the data to a CSV file
    csv_filename = 'party_wise_election_results_S18.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}.")
    
    # Display the DataFrame
    print(df.head())  # Print the first few rows of the DataFrame to check the data
    
    # Perform analysis
    perform_analysis(df)
    
    return df

# Function to perform analysis on the DataFrame
def perform_analysis(df):
    print("\nBasic Statistics")
    print(df.describe())

    total_seats_won = df['Seats Won'].sum()
    print(f"\nTotal seats won by all parties: {total_seats_won}")

    # Party with the most seats won
    max_seats = df['Seats Won'].max()
    party_max_seats = df[df['Seats Won'] == max_seats]['Party'].values[0]
    print(f"The party with the most seats won is {party_max_seats} with {max_seats} seats.")

    # Party with the least seats won
    min_seats = df['Seats Won'].min()
    party_min_seats = df[df['Seats Won'] == min_seats]['Party'].values[0]
    print(f"The party with the least seats won is {party_min_seats} with {min_seats} seats.")

    # Add percentage of total seats
    df['Percentage of Total Seats'] = (df['Seats Won'] / total_seats_won) * 100

    print("\nPercentage of total seats won by each party:")
    print(df[['Party', 'Percentage of Total Seats']])

    # Highest percentage of seats won
    max_percentage = df['Percentage of Total Seats'].max()
    party_max_percentage = df[df['Percentage of Total Seats'] == max_percentage]['Party'].values[0]
    print(f"\nThe party with the highest percentage of total seats won is {party_max_percentage} with {max_percentage:.2f}% of the seats.")

    # Analysis of seats leading
    total_leading_seats = df['Seats Leading'].sum()
    print(f"\nTotal seats leading by all parties: {total_leading_seats}")

    # Party with the maximum seats leading
    max_leading_seats = df['Seats Leading'].max()
    party_max_leading = df[df['Seats Leading'] == max_leading_seats]['Party'].values[0]
    print(f"The party with the most seats leading is {party_max_leading} with {max_leading_seats} seats.")

    # Party with the least seats leading
    min_leading_seats = df['Seats Leading'].min()
    party_min_leading = df[df['Seats Leading'] == min_leading_seats]['Party'].values[0]
    print(f"The party with the least seats leading is {party_min_leading} with {min_leading_seats} seats.")

    # Pie Chart for Seats Won
    plt.figure(figsize=(10, 8))
    plt.pie(df['Seats Won'], labels=df['Party'], autopct='%1.1f%%', startangle=140)
    plt.title('Election Results Distribution')
    plt.savefig('party_wise_pie_chart_S18.png')
    plt.show()

    # Bar Chart for Number of Seats Won
    plt.figure(figsize=(14, 7))
    plt.bar(df['Party'], df['Seats Won'], color='skyblue')
    plt.title('Number of Seats Won by Each Party')
    plt.xlabel('Party')
    plt.ylabel('Number of Seats')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('party_wise_bar_chart_S18.png')
    plt.show()

# Define the URL
url = 'https://results.eci.gov.in/AcResultGenJune2024/partywiseresult-S18.htm'

# Fetch data from the party-wise results page
fetch_party_wise_results(url)


# In[42]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

def fetch_by_election_results(url):
    print(f"Fetching data from {url}...")
    
    # Fetch the HTML content of the page
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully accessed the page: {url}")
    else:
        print(f"Failed to access the page: {url}. Status code: {response.status_code}")
        return None  # Return None if page access fails
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    print("HTML content fetched and parsed.")
    
    # Extract data from the appropriate elements
    const_boxes = soup.find_all('div', class_='const-box')
    
    if not const_boxes:
        print("No constituency boxes found.")
        return None
    
    print(f"Found {len(const_boxes)} constituency boxes.")
    
    # Initialize lists to store extracted data
    constituencies = []
    states = []
    results = []
    candidates = []
    parties = []

    for box in const_boxes:
        try:
            # Extracting data based on the HTML structure
            constituency = box.find('div', class_='box-content').find('h3').text.strip()
            state = box.find('div', class_='box-content').find('h4').text.strip()
            result = box.find('div', class_='box-content').find('h2').text.strip()
            candidate = box.find('div', class_='box-content').find('h5').text.strip()
            party = box.find('div', class_='box-content').find('h6').text.strip()
            
            constituencies.append(constituency)
            states.append(state)
            results.append(result)
            candidates.append(candidate)
            parties.append(party)
        except AttributeError:
            print("Some elements are missing in the div box.")
            continue

    # Check if data was extracted
    if not constituencies:
        print("No data extracted from the page.")
        return None
    
    print(f"Extracted data for {len(constituencies)} constituencies.")
    
    # Create a DataFrame to store the extracted data
    df = pd.DataFrame({
        'Constituency': constituencies,
        'State': states,
        'Result': results,
        'Candidate': candidates,
        'Party': parties
    })

    # Check if the DataFrame is populated correctly
    print("Data sample:")
    print(df.head())

    # Save the data to a CSV file
    csv_filename = 'bye_election_results_June2024.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}.")
    
    # Perform analysis
    perform_by_election_analysis(df)
    
    return df

def perform_by_election_analysis(df):
    print("\nBasic Statistics")
    print(df.describe(include='object'))

    # Analyze the distribution of results
    result_counts = df['Result'].value_counts()
    print("\nDistribution of Election Results:")
    print(result_counts)

    # Analyze the number of seats won by each party
    party_counts = df['Party'].value_counts()
    print("\nNumber of Seats Won by Each Party:")
    print(party_counts)

    # Add a column for counting the number of seats won by each party
    df['Seats Won'] = df['Result'].apply(lambda x: 1 if x == 'WON' else 0)
    seats_by_party = df.groupby('Party')['Seats Won'].sum().sort_values(ascending=False)

    # Total seats won by all parties
    total_seats_won = seats_by_party.sum()
    print(f"\nTotal seats won by all parties: {total_seats_won}")

    if seats_by_party.empty or total_seats_won == 0:
        print("Not enough data for plotting.")
        return  # Exit the function if there is no data to plot

    # Party with the most seats won
    max_seats = seats_by_party.max()
    party_max_seats = seats_by_party[seats_by_party == max_seats].index[0]
    print(f"\nThe party with the most seats won is {party_max_seats} with {max_seats} seats.")

    # Party with the least seats won
    min_seats = seats_by_party.min()
    party_min_seats = seats_by_party[seats_by_party == min_seats].index[0]
    print(f"\nThe party with the least seats won is {party_min_seats} with {min_seats} seats.")

    # Pie Chart for Seats Won
    plt.figure(figsize=(8, 8))
    plt.pie(seats_by_party, labels=seats_by_party.index, autopct='%1.1f%%', startangle=140)
    plt.title('Bye-Election Results Distribution')
    plt.savefig('bye_election_pie_chart_June2024.png')
    plt.show()

    # Bar Chart for Number of Seats Won by Each Party
    plt.figure(figsize=(12, 6))
    seats_by_party.plot(kind='bar', color='skyblue')
    plt.title('Number of Seats Won by Each Party')
    plt.xlabel('Party')
    plt.ylabel('Number of Seats')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('bye_election_bar_chart_June2024.png')
    plt.show()

def fetch_data_from_multiple_urls(urls, fetch_function):
    for url in urls:
        df = fetch_function(url)
        if df is not None:  # Ensure we only save if df is valid
            # Save data to CSV
            csv_filename = url.split('/')[-2] + '_results.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}.")

# Define the URLs for bye-elections
bye_election_urls = [
    'https://results.eci.gov.in/AcResultByeJune2024/'
    # Add other valid URLs as needed
]

# Fetch data from the bye-election results pages
fetch_data_from_multiple_urls(bye_election_urls, fetch_by_election_results)


# In[45]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

def fetch_party_wise_results(url):
    print(f"Fetching data from {url}...")
    
    # Fetch the HTML content of the page
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully accessed the page: {url}")
    else:
        print(f"Failed to access the page: {url}. Status code: {response.status_code}")
        return None  # Return None if page access fails
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    print("HTML content fetched and parsed.")
    
    # Extract data from the table
    table = soup.find('table', class_='table')
    
    if not table:
        print("No table found on the page.")
        return None
    
    print("Table found.")
    
    # Extract table rows
    rows = table.find('tbody').find_all('tr')
    
    # Initialize lists to store extracted data
    parties = []
    seats_won = []
    leading_counts = []
    total_counts = []
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 4:
            party = cols[0].text.strip()
            won = int(cols[1].text.strip())
            leading = int(cols[2].text.strip())
            total = int(cols[3].text.strip())
            
            parties.append(party)
            seats_won.append(won)
            leading_counts.append(leading)
            total_counts.append(total)

    # Check if data was extracted
    if not parties:
        print("No data extracted from the table.")
        return None
    
    print(f"Extracted data for {len(parties)} parties.")
    
    # Create a DataFrame to store the extracted data
    df = pd.DataFrame({
        'Party': parties,
        'Seats Won': seats_won,
        'Leading': leading_counts,
        'Total Seats': total_counts
    })
    
    # Check if the DataFrame is populated correctly
    print("Data sample:")
    print(df.head())
    
    # Save the data to a CSV file
    csv_filename = 'party_wise_results_June2024.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}.")
    
    # Perform analysis
    perform_party_wise_analysis(df)
    
    return df

def perform_party_wise_analysis(df):
    print("\nBasic Statistics")
    print(df.describe(include='object'))

    # Analyze the distribution of seats won
    seats_distribution = df['Seats Won'].value_counts()
    print("\nDistribution of Seats Won:")
    print(seats_distribution)

    # Analyze the total number of seats won by each party
    party_seat_counts = df.groupby('Party')['Seats Won'].sum().sort_values(ascending=False)

    # Total seats won by all parties
    total_seats_won = party_seat_counts.sum()
    print(f"\nTotal seats won by all parties: {total_seats_won}")

    if party_seat_counts.empty or total_seats_won == 0:
        print("Not enough data for plotting.")
        return  # Exit the function if there is no data to plot

    # Party with the most seats won
    max_seats = party_seat_counts.max()
    party_max_seats = party_seat_counts[party_seat_counts == max_seats].index[0]
    print(f"\nThe party with the most seats won is {party_max_seats} with {max_seats} seats.")

    # Party with the least seats won
    min_seats = party_seat_counts.min()
    party_min_seats = party_seat_counts[party_seat_counts == min_seats].index[0]
    print(f"\nThe party with the least seats won is {party_min_seats} with {min_seats} seats.")

    # Pie Chart for Seats Won by Parties
    plt.figure(figsize=(10, 8))
    plt.pie(party_seat_counts, labels=party_seat_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Party-Wise Seat Distribution for June 2024 Elections')
    plt.savefig('party_wise_seat_distribution_June2024.png')
    plt.show()

    # Bar Chart for Number of Seats Won by Each Party
    plt.figure(figsize=(12, 8))
    party_seat_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Seats Won by Each Party in June 2024')
    plt.xlabel('Party')
    plt.ylabel('Number of Seats Won')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('party_wise_seats_bar_chart_June2024.png')
    plt.show()

def fetch_data_from_multiple_urls(urls, fetch_function):
    for url in urls:
        df = fetch_function(url)
        if df is not None:  # Ensure we only save if df is valid
            # Save data to CSV
            csv_filename = url.split('/')[-2] + '_results.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}.")

# Define the URL for the party-wise results
party_wise_results_url = [
    'https://results.eci.gov.in/AcResultGen2ndJune2024/partywiseresult-S02.htm'
]

# Fetch data from the party-wise results page
fetch_data_from_multiple_urls(party_wise_results_url, fetch_party_wise_results)


# In[46]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

def fetch_party_wise_results(url):
    print(f"Fetching data from {url}...")
    
    # Fetch the HTML content of the page
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully accessed the page: {url}")
    else:
        print(f"Failed to access the page: {url}. Status code: {response.status_code}")
        return None  # Return None if page access fails
    
    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    print("HTML content fetched and parsed.")
    
    # Extract data from the table
    table = soup.find('table', class_='table')
    
    if not table:
        print("No table found on the page.")
        return None
    
    print("Table found.")
    
    # Extract table rows
    rows = table.find('tbody').find_all('tr')
    
    # Initialize lists to store extracted data
    parties = []
    seats_won = []
    leading_counts = []
    total_counts = []
    
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 4:
            party = cols[0].text.strip()
            won = int(cols[1].text.strip())
            leading = int(cols[2].text.strip())
            total = int(cols[3].text.strip())
            
            parties.append(party)
            seats_won.append(won)
            leading_counts.append(leading)
            total_counts.append(total)

    # Check if data was extracted
    if not parties:
        print("No data extracted from the table.")
        return None
    
    print(f"Extracted data for {len(parties)} parties.")
    
    # Create a DataFrame to store the extracted data
    df = pd.DataFrame({
        'Party': parties,
        'Seats Won': seats_won,
        'Leading': leading_counts,
        'Total Seats': total_counts
    })
    
    # Check if the DataFrame is populated correctly
    print("Data sample:")
    print(df.head())
    
    # Save the data to a CSV file
    csv_filename = 'party_wise_results_Sikkim_2024.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}.")
    
    # Perform analysis
    perform_party_wise_analysis(df)
    
    return df

def perform_party_wise_analysis(df):
    print("\nBasic Statistics")
    print(df.describe(include='object'))

    # Analyze the distribution of seats won
    seats_distribution = df['Seats Won'].value_counts()
    print("\nDistribution of Seats Won:")
    print(seats_distribution)

    # Analyze the total number of seats won by each party
    party_seat_counts = df.groupby('Party')['Seats Won'].sum().sort_values(ascending=False)

    # Total seats won by all parties
    total_seats_won = party_seat_counts.sum()
    print(f"\nTotal seats won by all parties: {total_seats_won}")

    if party_seat_counts.empty or total_seats_won == 0:
        print("Not enough data for plotting.")
        return  # Exit the function if there is no data to plot

    # Party with the most seats won
    max_seats = party_seat_counts.max()
    party_max_seats = party_seat_counts[party_seat_counts == max_seats].index[0]
    print(f"\nThe party with the most seats won is {party_max_seats} with {max_seats} seats.")

    # Party with the least seats won
    min_seats = party_seat_counts.min()
    party_min_seats = party_seat_counts[party_seat_counts == min_seats].index[0]
    print(f"\nThe party with the least seats won is {party_min_seats} with {min_seats} seats.")

    # Pie Chart for Seats Won by Parties
    plt.figure(figsize=(10, 8))
    plt.pie(party_seat_counts, labels=party_seat_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Party-Wise Seat Distribution for Sikkim 2024 Elections')
    plt.savefig('party_wise_seat_distribution_Sikkim_2024.png')
    plt.show()

    # Bar Chart for Number of Seats Won by Each Party
    plt.figure(figsize=(12, 8))
    party_seat_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Seats Won by Each Party in Sikkim 2024')
    plt.xlabel('Party')
    plt.ylabel('Number of Seats Won')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('party_wise_seats_bar_chart_Sikkim_2024.png')
    plt.show()

def fetch_data_from_multiple_urls(urls, fetch_function):
    for url in urls:
        df = fetch_function(url)
        if df is not None:  # Ensure we only save if df is valid
            # Save data to CSV
            csv_filename = url.split('/')[-2] + '_results.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}.")

# Define the URL for the party-wise results
party_wise_results_url = [
    'https://results.eci.gov.in/AcResultGen2ndJune2024/partywiseresult-S21.htm'
]

# Fetch data from the party-wise results page
fetch_data_from_multiple_urls(party_wise_results_url, fetch_party_wise_results)


# In[ ]:




