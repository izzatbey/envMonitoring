import subprocess
import time
import pandas as pd
import plotly.express as px

def collect_tegrastats_data(duration=60, interval=5):
    timestamps = []
    power_values = []

    print("Starting tegrastats...")
    process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    start_time = time.time()
    while time.time() - start_time < duration:
        output = process.stdout.readline().strip()
        if not output:
            continue
        print(f"tegrastats output:\n{output}")  # Debugging output
        
        # Parse the power value
        power = parse_power(output)
        if power is not None:
            timestamps.append(time.time())
            power_values.append(power)
            print(f"Time: {timestamps[-1]}, Power: {power_values[-1]} mW")  # Debugging output
        
        time.sleep(interval)

    process.terminate()
    return pd.DataFrame({'Timestamp': timestamps, 'Power': power_values})

def parse_power(tegrastats_output):
    try:
        tokens = tegrastats_output.split()
        
        # Print full output for debugging
        print("Full tegrastats output:\n{}".format(tegrastats_output))

        index_vdd_in = tokens.index('VDD_IN') if 'VDD_IN' in tokens else -1
        
        if index_vdd_in != -1:
            power_str = tokens[index_vdd_in + 1].strip('/')
            if '/' in power_str:
                power_values = power_str.split('/')
                if len(power_values) >= 1:
                    power_str = power_values[0]
            print("Parsed power string: {}".format(power_str))
            
            # Convert power string to float
            power_value = float(power_str.replace('mW', ''))
            return power_value
    except Exception as e:
        print(f"Error parsing power data: {e}")
    return None

def main():
    print("Start data collection")
    # Collect power consumption data for 60 seconds with 5-second intervals
    df = collect_tegrastats_data(duration=60, interval=5)

    print("Collected data:")  # Debugging output
    print(df)  # Debugging output

    if not df.empty:
        # Convert timestamps to human-readable format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

        # Create Plotly figure
        fig = px.line(df, x='Timestamp', y='Power', title='Nvidia Jetson Power Consumption Over Time', labels={'Power': 'Power (mW)'})

        # Save the plot to a file
        fig.write_image("power_consumption_plot.png")

        # Show the plot
        fig.show()
    else:
        print("No data collected")

if __name__ == "__main__":
    print("Main start")
    main()
