import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def plot_performance():
    # 1. Caminhos
    current_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Ajuste o caminho para onde os CSVs estão sendo salvos
    # Se estiverem em 'results/', use 'results'
    # Se estiverem na raiz, use '..'
    results_dir = os.path.join(current_folder, "..", "results")
    plots_dir = os.path.join(current_folder, "..", "results", "graphs")

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # 2. Buscar CSVs
    search_path = os.path.join(results_dir, "results_*.csv")
    files = glob.glob(search_path)

    if not files:
        print(f"Nenhum CSV encontrado em {results_dir}")
        return

    data_frames = []

    # 3. Processar cada modelo
    for f in files:
        try:
            df = pd.read_csv(f, sep="#")
            
            # Extrai o nome do modelo do nome do arquivo
            filename = os.path.basename(f)
            model_name = filename.replace("results_", "").replace(".csv", "")
            
            df["model"] = model_name
            data_frames.append(df)
        except Exception as e:
            print(f"Erro ao processar arquivo {f}: {e}")

    if not data_frames:
        print("Nenhum dado válido extraído dos CSVs.")
        return

    full_df = pd.concat(data_frames, ignore_index=True)

    # Mapeamento de colunas para nomes legíveis
    metric_map = {
        "metric_bertscore_f1": "BERTScore (F1)",
        "metric_g_eval_score": "G-Eval (Adherence)",
        "metric_g_eval_quality_score": "G-Eval (Quality)",
        "metric_len_ratio": "Length Ratio",
        "tempo_total": "Time (s)",
        "total_tokens": "Total Tokens"
    }
    
    # Filtra apenas as colunas que existem no DF
    available_metrics = [col for col in metric_map.keys() if col in full_df.columns]
    
    if not available_metrics:
        print("Nenhuma métrica encontrada para plotar.")
        return

    # Configuração visual
    sns.set_theme(style="whitegrid")
    
    # Agrupa por modelo e calcula média
    means = full_df.groupby("model")[available_metrics].mean().reset_index()
    
    # --- Gráficos de Barras (Médias por Modelo) ---
    for metric_col in available_metrics:
        plt.figure(figsize=(10, 6))
        readable_name = metric_map.get(metric_col, metric_col)
        
        # Ordena para ficar bonito
        sorted_means = means.sort_values(by=metric_col, ascending=False)
        
        ax = sns.barplot(x="model", y=metric_col, data=sorted_means, palette="viridis")
        
        plt.title(f"Average {readable_name} by Model")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(readable_name)
        plt.xlabel("Model")
        
        # Adiciona valores nas barras
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        filename = f"bar_{metric_col}.png"
        plt.savefig(os.path.join(plots_dir, filename))
        plt.close()
        print(f"Gerado: {filename}")

    # --- Scatter Plot: Qualidade vs Custo (Tokens) ---
    if "total_tokens" in full_df.columns and "metric_g_eval_quality_score" in full_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=means, 
            x="total_tokens", 
            y="metric_g_eval_quality_score", 
            hue="model", 
            style="model", 
            s=100
        )
        plt.title("Efficiency: Quality vs Token Usage")
        plt.xlabel("Average Total Tokens")
        plt.ylabel("Average G-Eval (Quality)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, "scatter_quality_vs_tokens.png"))
        plt.close()
        print("Gerado: scatter_quality_vs_tokens.png")

    # --- Scatter Plot: Qualidade vs Tempo ---
    if "tempo_total" in full_df.columns and "metric_g_eval_quality_score" in full_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=means, 
            x="tempo_total", 
            y="metric_g_eval_quality_score", 
            hue="model", 
            style="model", 
            s=100
        )
        plt.title("Efficiency: Quality vs Time")
        plt.xlabel("Average Time (s)")
        plt.ylabel("Average G-Eval (Quality)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(plots_dir, "scatter_quality_vs_time.png"))
        plt.close()
        print("Gerado: scatter_quality_vs_time.png")

    print(f"\nSucesso! Gráficos gerados em: {plots_dir}")

if __name__ == "__main__":
    plot_performance()
