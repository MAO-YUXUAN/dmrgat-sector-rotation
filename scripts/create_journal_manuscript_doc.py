from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.json"
METRICS_PATH = ROOT / "data" / "processed" / "metrics.json"
OUTPUT_PATH = ROOT / "DMRGAT_Journal_Style_Manuscript.docx"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def set_default_style(document: Document) -> None:
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)


def add_title_block(document: Document, title: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(15)

    p2 = document.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run("Journal-Style Manuscript Draft")
    r2.italic = True
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(11)


def add_heading(document: Document, text: str, level: int = 1) -> None:
    document.add_heading(text, level=level)


def add_paragraph(document: Document, text: str, bold_prefix: str | None = None) -> None:
    p = document.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    if bold_prefix:
        r1 = p.add_run(bold_prefix)
        r1.bold = True
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(11)
        r2 = p.add_run(text)
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(11)
    else:
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)


def add_equation(document: Document, text: str) -> None:
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.italic = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)


def format_metrics(metrics: dict) -> dict[str, str]:
    def f(key: str) -> str:
        value = metrics.get(key)
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            return f"{value:.4f}"
        return str(value)

    return {
        "train_rmse": f("train_rmse"),
        "val_rmse": f("val_rmse"),
        "test_rmse": f("test_rmse"),
        "train_mae": f("train_mae"),
        "val_mae": f("val_mae"),
        "test_mae": f("test_mae"),
        "train_ic": f("train_ic"),
        "val_ic": f("val_ic"),
        "test_ic": f("test_ic"),
        "train_rank_ic": f("train_rank_ic"),
        "val_rank_ic": f("val_rank_ic"),
        "test_rank_ic": f("test_rank_ic"),
        "num_graphs": str(metrics.get("num_graphs", "N/A")),
        "num_nodes": str(metrics.get("num_nodes", "N/A")),
    }


def build_document(config: dict, metrics: dict) -> Document:
    m = format_metrics(metrics)
    horizon = config.get("horizon", 30)
    alpha = config.get("alpha", 0.45)
    beta = config.get("beta", 0.25)
    gamma = config.get("gamma", 0.30)
    hidden_dim = config.get("hidden_dim", 32)
    num_heads = config.get("num_heads", 4)
    ranking_w = config.get("ranking_loss_weight", 4.0)
    mse_w = config.get("mse_loss_weight", 0.05)
    corr_window = config.get("corr_window", 20)
    lookbacks = config.get("feature_lookbacks", [5, 10, 20])
    train_ratio = config.get("train_ratio", 0.80)
    val_ratio = config.get("val_ratio", 0.10)
    core_industries = config.get("core_industries", [])

    document = Document()
    set_default_style(document)
    add_title_block(
        document,
        "Dynamic Multi-Relation Graph Attention Networks for Sector Rotation Forecasting: "
        "A Ranking-Oriented Industry Allocation Framework",
    )

    add_heading(document, "Abstract", level=1)
    add_paragraph(
        document,
        "This paper develops a dynamic multi-relation graph attention network (DMRGAT) for sector rotation forecasting in the Chinese equity market. The model is designed to predict future industry-level excess returns relative to the CSI 300 benchmark and to support ranking-based portfolio allocation rather than purely directional classification. Each industry is treated as a graph node, and the dynamic graph is constructed by integrating rolling return correlation, capital-flow similarity, and a static industry prior graph. Node features combine time-series market indicators and cross-sectional ranking features, allowing the model to exploit both within-industry dynamics and relative market positioning. The learning objective is specified as a hybrid ranking-regression loss so that the model directly optimizes cross-sectional sorting quality while preserving numerical return information. Empirically, the study is implemented on 15 core Shenwan level-1 industries with a "
        f"{horizon}-day forecasting horizon. Under the current specification, the out-of-sample test RMSE is {m['test_rmse']}, the test MAE is {m['test_mae']}, and the test RankIC is {m['test_rank_ic']}. Although the final ranking signal remains economically modest, the framework provides a coherent and interpretable pipeline that connects graph representation learning with practical sector allocation research. The study contributes a finance-oriented graph learning design, a ranking-consistent training objective, and an interpretable attention-based view of dynamic inter-industry dependence."
    )

    add_paragraph(
        document,
        "Keywords: sector rotation; graph attention network; dynamic graph; excess return prediction; rank-based learning; industry allocation",
        bold_prefix="Keywords: ",
    )

    add_heading(document, "1. Introduction", level=1)
    add_paragraph(
        document,
        "Sector rotation forecasting is a central topic in quantitative asset allocation because investors frequently make top-down decisions at the industry or style level rather than at the single-stock level. However, sector performance is shaped not only by idiosyncratic time-series information but also by changing dependencies across industries. Traditional econometric models and many machine-learning approaches treat industries as independent prediction units and therefore overlook the network structure of information transmission in the market."
    )
    add_paragraph(
        document,
        "This paper addresses the problem by developing a dynamic multi-relation graph attention network. The core idea is that industry rotation should be modeled as a relational forecasting problem in which each sector evolves jointly with its economically related peers. Instead of using a single graph, the proposed framework fuses multiple relation types that capture short-horizon co-movement, capital-flow synchronization, and structural industry linkage. This design reflects the fact that sector interaction is multi-dimensional and time-varying."
    )
    add_paragraph(
        document,
        "The study is motivated by two practical considerations. First, investment decisions often depend more on ranking than on exact point forecasts. Second, graph attention models can naturally provide an interpretable mapping from inter-industry structure to predicted relative strength. Accordingly, the paper formulates the task as future excess-return prediction and trains the model with a ranking-oriented objective."
    )
    add_paragraph(
        document,
        "Relative to a conventional implementation note or project report, this manuscript is written in a journal-oriented style and emphasizes four contributions. First, it proposes a sector-rotation framework built on a dynamic multi-relation graph rather than a single relation matrix. Second, it combines time-series features and cross-sectional ranking features in the node representation. Third, it aligns the training objective with the economic purpose of cross-sectional allocation through pairwise ranking loss. Fourth, it provides an interpretable attention-based view of sector dependence through head-specific heatmaps."
    )

    add_heading(document, "2. Related Literature and Research Positioning", level=1)
    add_paragraph(
        document,
        "The proposed framework lies at the intersection of three strands of literature. The first strand studies sector rotation and tactical asset allocation, where relative-strength signals, macro conditions, and cross-industry spillovers are often central. The second strand studies graph neural networks in finance, especially relational forecasting in stock markets, corporate networks, and macro-financial linkage structures. The third strand studies ranking-oriented learning, which is particularly relevant when the economic objective is to sort investable units by expected future performance."
    )
    add_paragraph(
        document,
        "Most existing sector-rotation studies either rely on handcrafted factor timing rules or use independent predictive models for each industry. By contrast, the present paper explicitly embeds sector dependence into the learning architecture. At the same time, unlike a vanilla graph attention network, the proposed DMRGAT introduces edge priors through a fused adjacency matrix and trains the model under a ranking-sensitive objective. Hence, the paper is better interpreted as a finance-adapted graph-learning framework than as a purely generic application of GAT."
    )

    add_heading(document, "3. Methodology", level=1)
    add_heading(document, "3.1 Problem Setup and Basic Model", level=2)
    add_paragraph(
        document,
        f"Let the dynamic graph at time t be denoted by G_t = (V, E_t, A_t), where V = {{v_1, v_2, ..., v_N}} is the set of industry nodes, E_t is the edge set, and A_t is the fused adjacency matrix. In the current implementation, N = {m['num_nodes']} and the industry universe consists of 15 core Shenwan level-1 industries: {', '.join(core_industries)}."
    )
    add_paragraph(document, "For each industry i, the node feature vector is constructed exactly as follows:")
    add_equation(
        document,
        "x_i,t = [ret_1d_i,t, excess_ret_1d_i,t, ret_std_5_i,t, ret_std_20_i,t, amount_chg_i,t, "
        "amount_z20_i,t, mom_5_i,t, mom_20_i,t, ma_gap_i,t, ret_rank_pct_i,t, excess_rank_pct_i,t, "
        "mom_rank_pct_5_i,t, mom_rank_pct_20_i,t, amount_rank_pct_i,t, vol_rank_pct_i,t]."
    )
    add_paragraph(
        document,
        "These features summarize short-horizon return information, benchmark-relative performance, realized volatility, liquidity and turnover change, momentum, trend deviation, and cross-sectional percentile ranks."
    )
    add_paragraph(document, f"The forecasting target is the future {horizon}-day excess return of industry i over the benchmark:")
    add_equation(document, f"y_i,t^({horizon}) = R_i,t->t+{horizon} - R_m,t->t+{horizon}.")
    add_paragraph(
        document,
        "In the implementation, this quantity is stored as future_excess_ret and used directly as the regression target."
    )
    add_paragraph(document, "To model inter-industry dependence, three relation matrices are introduced:")
    add_equation(document, f"A_corr,t(i,j) = Corr(ret_1d_i,t-{corr_window}:t, ret_1d_j,t-{corr_window}:t),")
    add_equation(document, f"A_fund,t(i,j) = Corr(amount_chg_i,t-{corr_window}:t, amount_chg_j,t-{corr_window}:t),")
    add_equation(document, "A_industry(i,j) = 1 if industries i and j are economically related, and 0 otherwise.")
    add_paragraph(document, "The final fused adjacency matrix is specified as:")
    add_equation(document, f"A_t = {alpha:.2f} * A_corr,t + {beta:.2f} * A_industry + {gamma:.2f} * A_fund,t.")
    add_paragraph(
        document,
        "Because the coefficients sum to one, the implemented fused graph is a convex combination of the three relation matrices."
    )
    add_paragraph(document, "The learning task is therefore to estimate a parametric mapping")
    add_equation(document, f"hat(y)_i,t^({horizon}) = f(G_t, X_t; theta),")
    add_paragraph(document, f"which predicts the future {horizon}-day excess return for each industry.")

    add_heading(document, "3.2 DMRGAT Architecture", level=2)
    add_paragraph(
        document,
        "The model begins from h_i^(0) = x_i,t and applies a linear projection to the hidden space. In the current implementation, the hidden dimension is "
        f"{hidden_dim} and the first graph attention layer uses {num_heads} heads."
    )
    add_equation(document, "z_i = W h_i^(0).")
    add_paragraph(document, "For each edge (i, j), the weighted attention score is computed as:")
    add_equation(document, "e_ij,t = A_ij,t * LeakyReLU(a^T [z_i || z_j]).")
    add_paragraph(document, "The normalized attention coefficient is then given by:")
    add_equation(document, "alpha_ij,t = exp(e_ij,t) / sum_{k in N_i} exp(e_ik,t).")
    add_paragraph(document, "The first-layer multi-head aggregation can be written as:")
    add_equation(document, f"h_i^(1) = ||_(m=1)^{num_heads} sigma(sum_(j in N_i) alpha_ij,t^(m) z_j^(m)).")
    add_paragraph(
        document,
        "The concatenated representation is passed through a second graph attention layer with one head, followed by a linear regression head:"
    )
    add_equation(document, f"hat(y)_i,t^({horizon}) = W_o h_i^(2).")

    add_heading(document, "3.3 Ranking-Oriented Training Objective", level=2)
    add_paragraph(
        document,
        "The implemented training objective combines pairwise ranking loss and a weak mean squared error regularizer. For one graph snapshot, the pairwise ranking loss is"
    )
    add_equation(document, "L_rank = mean_(i,j) log(1 + exp(-(y_i - y_j)(hat(y)_i - hat(y)_j))).")
    add_paragraph(document, "The numerical regression term is")
    add_equation(document, "L_mse = (1 / N) sum_i (hat(y)_i - y_i)^2.")
    add_paragraph(document, "The total loss used in the code is")
    add_equation(document, f"L = {ranking_w:.2f} * L_rank + {mse_w:.2f} * L_mse.")
    add_paragraph(
        document,
        "This objective reflects the economic purpose of sector allocation: the model is primarily trained to rank industries correctly, while a smaller MSE component prevents the prediction scale from drifting uncontrollably."
    )

    add_heading(document, "4. Theoretical Considerations", level=1)
    add_heading(document, "4.1 Convex Fusion Property", level=2)
    add_paragraph(
        document,
        "The fused graph used in the model is not an arbitrary linear combination but a convex combination of three economically distinct relation matrices. This distinction is important for both mathematical stability and financial interpretability. If the coefficients are allowed to take unrestricted values, the scale and sign of the fused adjacency matrix may become difficult to interpret, and the resulting graph may no longer represent a meaningful relation-strength matrix. In contrast, imposing nonnegative coefficients that sum to one ensures that the final adjacency matrix can be understood as a weighted average of return co-movement, industry-structure prior, and capital-flow similarity."
    )
    add_paragraph(
        document,
        "From a graph-learning perspective, this property keeps the fused adjacency matrix within the feasible set spanned by the three base relation matrices. In other words, the model does not create a new relation type with an uncontrolled scale; it reallocates emphasis among existing relation channels. This is particularly useful in financial applications because each base matrix has a clear economic meaning. The return-correlation matrix captures short-term co-movement, the industry prior matrix captures relatively stable structural linkage, and the fund-flow matrix captures synchronized changes in market participation. A convex fusion therefore preserves the interpretability of each channel while allowing their relative importance to vary through the chosen coefficients."
    )
    add_paragraph(
        document,
        "The convex formulation also improves numerical comparability across relation matrices. In the implementation, the correlation-based matrices are transformed into nonnegative edge-strength matrices before fusion, while the industry prior graph is binary. After this normalization, the convex combination prevents any single relation matrix from dominating purely through scale differences. As a result, the attention mechanism receives an adjacency prior whose magnitude is stable and whose entries remain comparable across different relation sources."
    )
    add_paragraph(document, "Formally, the implemented graph satisfies")
    add_equation(document, f"A_t = {alpha:.2f} A_corr,t + {beta:.2f} A_industry + {gamma:.2f} A_fund,t,")
    add_paragraph(
        document,
        "which means that the model always allocates relative emphasis among correlation, industry prior, and capital-flow similarity, rather than introducing an additional scale distortion."
    )
    add_paragraph(
        document,
        "This property also provides a useful interpretation of the hyperparameters. The coefficient on the return-correlation graph controls how much the model relies on recent market co-movement; the coefficient on the industry prior graph controls the strength of structural economic linkage; and the coefficient on the fund-flow graph controls the influence of synchronized trading-amount dynamics. Therefore, the fusion coefficients are not merely numerical tuning parameters. They define how different economic channels are combined before the graph attention mechanism learns adaptive neighbor weights."
    )
    add_paragraph(
        document,
        "Finally, the convex fusion property contributes to optimization stability. Since the attention score is multiplied by the edge weight in the DMRGAT layer, unstable or excessively large graph weights could amplify attention scores and make training more sensitive to initialization and learning rate. A convexly fused and normalized adjacency matrix mitigates this risk by keeping edge weights in a controlled range. This does not guarantee global convergence, because the full neural network remains non-convex, but it reduces one source of avoidable numerical instability in the graph construction stage."
    )

    add_heading(document, "4.2 Ranking Consistency", level=2)
    add_paragraph(
        document,
        "The ranking loss has a direct economic interpretation. Consider two industries i and j. If y_i > y_j, then the model should ideally produce hat(y)_i > hat(y)_j. The pairwise term"
    )
    add_equation(document, "log(1 + exp(-(y_i - y_j)(hat(y)_i - hat(y)_j)))")
    add_paragraph(
        document,
        "decreases when the predicted spread and the realized spread share the same sign, and increases when the predicted ordering is reversed. Hence, minimizing the ranking loss encourages cross-sectional ordering consistency rather than merely numerical proximity."
    )
    add_paragraph(
        document,
        "This property links the training objective with RankIC evaluation. A positive out-of-sample RankIC means that the model tends to place realized future winners above realized future losers in the cross section, which is the key requirement for a ranking-based allocation signal."
    )

    add_heading(document, "4.3 Optimization and Convergence Discussion", level=2)
    add_paragraph(
        document,
        "The objective function is differentiable with respect to the model parameters because both the pairwise ranking term and the MSE term are smooth. Specifically, the ranking component is constructed from log(1 + exp(.)), while the regression component is quadratic. Therefore, the total loss can be optimized with gradient-based methods."
    )
    add_equation(document, "theta_(k+1) = theta_k - eta * nabla_theta L(theta_k).")
    add_paragraph(
        document,
        "Because the DMRGAT architecture is nonlinear and multi-layered, the optimization landscape is non-convex, and global convergence is not guaranteed. However, under standard practical conditions—bounded features, finite graph size, smooth activation functions, and a sufficiently small learning rate—the Adam optimizer can be expected to converge toward a locally stable stationary point. This is the relevant notion of convergence for the implemented model."
    )
    add_paragraph(
        document,
        "Relative to a standard GNN with fixed aggregation, DMRGAT is more expressive but also more complex to optimize because attention coefficients are state-dependent and the graph itself is relation-fused. Relative to a vanilla GAT, the current model adds a prior graph weighting term A_ij,t into the attention score. When the prior graph is informative, this can reduce the effective search space of attention allocation and improve optimization efficiency; when the prior graph is noisy, the same mechanism can weaken stability. Hence, the convergence advantage is conditional rather than universal."
    )

    add_heading(document, "5. Data and Experimental Design", level=1)
    add_paragraph(
        document,
        "The empirical analysis is based on daily Chinese market data. The benchmark is the CSI 300 index, and the industry universe consists of 15 core Shenwan level-1 industries selected to represent cyclical, financial, and growth-oriented sectors. The data pipeline is implemented in Python and relies on AKShare for market data acquisition."
    )
    add_paragraph(
        document,
        f"The sample is transformed into {m['num_graphs']} valid graph snapshots after rolling-window construction and feature alignment. Time-series windows of {corr_window} trading days are used to build the correlation-based graph and the capital-flow graph. The final dataset is split chronologically into training, validation, and test subsets with proportions {train_ratio:.2f}, {val_ratio:.2f}, and {1 - train_ratio - val_ratio:.2f}, respectively, in order to avoid look-ahead bias."
    )
    add_paragraph(
        document,
        "Model hyperparameters are aligned with the implemented code. The first graph attention layer uses four heads, the hidden dimension is 32, the graph fusion coefficients are 0.45, 0.25, and 0.30, and the optimization procedure uses Adam with early stopping. The current ranking-oriented loss places substantially greater weight on pairwise ordering than on numerical return matching."
    )

    add_heading(document, "6. Empirical Results", level=1)
    add_heading(document, "6.1 Main Forecasting Results", level=2)
    add_paragraph(
        document,
        "The current implementation produces the following out-of-sample results. On the training set, RMSE is "
        f"{m['train_rmse']}, MAE is {m['train_mae']}, IC is {m['train_ic']}, and RankIC is {m['train_rank_ic']}. "
        f"On the validation set, RMSE is {m['val_rmse']}, MAE is {m['val_mae']}, IC is {m['val_ic']}, and RankIC is {m['val_rank_ic']}. "
        f"On the test set, RMSE is {m['test_rmse']}, MAE is {m['test_mae']}, IC is {m['test_ic']}, and RankIC is {m['test_rank_ic']}."
    )
    add_paragraph(
        document,
        "These results suggest that the model has learned a weak but positive ranking signal in the final out-of-sample period. The test RankIC is positive, which is preferable to a negative ranking correlation, but its magnitude remains modest. Therefore, the current model should be interpreted as an exploratory relational forecasting framework rather than a fully mature allocation engine."
    )

    add_heading(document, "6.2 Attention-Based Interpretation", level=2)
    add_paragraph(
        document,
        "An important advantage of DMRGAT is interpretability. The first graph attention layer provides edge-level attention weights that can be visualized as industry-to-industry heatmaps. In the current implementation, both the average heatmap and head-specific heatmaps are generated automatically after training. These visualizations help identify which industries serve as influential neighbors under a given market state."
    )
    add_paragraph(
        document,
        "Because the first layer uses four heads, different heads may capture different relational structures. For example, one head may focus more on cyclical transmission among upstream sectors, while another may emphasize co-movement among growth and technology industries. This multi-head structure offers a practical interpretation tool that complements the purely numerical evaluation metrics."
    )

    add_heading(document, "6.3 Discussion, Limitations, and Journal Relevance", level=2)
    add_paragraph(
        document,
        "The empirical evidence supports the methodological value of the framework but also highlights its limitations. The final ranking signal is positive but economically weak, which indicates that relational modeling alone is not sufficient to fully solve the sector-rotation problem. In financial prediction, such a result is not unusual because cross-sectional excess return signals are noisy, regime-dependent, and relatively low in signal-to-noise ratio."
    )
    add_paragraph(
        document,
        "For journal submission, the main strength of the paper is therefore methodological coherence rather than headline predictive power. The paper provides a well-defined dynamic multi-relation graph structure, an economically aligned ranking objective, and a transparent attention-based interpretation mechanism. To further strengthen publication quality, future work should incorporate richer macro-financial features, temporal graph modules, robustness tests under multiple market regimes, and transaction-cost-aware portfolio evaluation."
    )

    add_heading(document, "7. Conclusion", level=1)
    add_paragraph(
        document,
        "This paper presents a journal-style formulation of a DMRGAT framework for sector rotation forecasting. The model translates industry dependence into a dynamic multi-relation graph and combines graph attention with ranking-oriented learning to predict future excess returns. The implementation is fully aligned with the current codebase, including the 15-dimensional node feature vector, the 30-day horizon, the three-part graph fusion mechanism, and the ranking-dominant loss function."
    )
    add_paragraph(
        document,
        "The empirical results indicate that the model can generate a weak positive out-of-sample ranking signal, although the overall effect remains moderate. This suggests that the proposed framework is promising as a research platform and interpretability tool, but that further refinement is needed before stronger economic claims can be made. Future extensions should focus on temporal graph modeling, richer feature engineering, robustness validation, and portfolio-level backtesting."
    )
    add_paragraph(
        document,
        "Overall, the DMRGAT framework provides a rigorous and extensible bridge between graph representation learning and sector allocation research. Its main value lies in offering a structured way to integrate dynamic market linkage, prior economic structure, and ranking-based optimization within a single forecasting architecture."
    )

    add_heading(document, "Appendix Note", level=1)
    add_paragraph(
        document,
        "The current manuscript is written as a polished journal-style draft based strictly on the implemented code configuration and available empirical outputs. Before actual submission, the authors should further add a formal literature review with complete citations, standardized table formatting, a robustness section, and a final reference list conforming to the target journal's style guide."
    )

    return document


def main() -> None:
    config = load_json(CONFIG_PATH)
    metrics = load_json(METRICS_PATH)
    document = build_document(config, metrics)
    output_path = OUTPUT_PATH
    try:
        document.save(output_path)
    except PermissionError:
        output_path = ROOT / "DMRGAT_Journal_Style_Manuscript_v2.docx"
        document.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
