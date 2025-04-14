import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_performance_data():
    df_normal = pd.read_csv('perf.csv', index_col='function')
    df_opt = pd.read_csv('perf_opt.csv', index_col='function')
    
    df_combined = pd.DataFrame({
        'Normal': df_normal['time'],
        'Optimized': df_opt['time']
    })
    
    return df_combined

def plot_total_performance(df):
    plt.figure(figsize=(10, 6))
    total_data = df.loc['total']
    
    plt.bar(['Normal', 'Optimized'], [total_data['Normal'], total_data['Optimized']])
    plt.title('Total Execution Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.savefig('total_performance.png')
    plt.close()

def plot_function_performance(df):
    df_functions = df.drop('total')
    
    plt.figure(figsize=(12, 6))
    df_functions.plot(kind='bar')
    plt.title('Performance by Function')
    plt.xlabel('Function')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('function_performance.png')
    plt.close()

def main():
    df = load_performance_data()
    plot_total_performance(df)
    plot_function_performance(df)
    
    print("Performance analysis completed. Check total_performance.png and function_performance.png")

if __name__ == "__main__":
    main()
