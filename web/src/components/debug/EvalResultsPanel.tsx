import { useEffect, useState } from "react";
import { getEvalResults } from "../../api";
import { METRIC_GROUPS } from "../../types";
import type { EvalResultFile, MetricGroup } from "../../types";

function fmt(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

const GROUP_LABELS: Record<MetricGroup, string> = {
  retrieval: "Retrieval",
  generation: "Generation",
  reward: "Reward",
  latency: "Latency",
  other: "Other",
};

/**
 * A file's numbers bucketed by side. Older backends send only the flat
 * `metrics`; those fall under "Other" rather than vanishing.
 */
function groupsOf(file: EvalResultFile): Array<[MetricGroup, Record<string, number>]> {
  const groups = file.groups ?? { other: file.metrics };
  return METRIC_GROUPS.flatMap((group) => {
    const values = groups[group];
    return values && Object.keys(values).length > 0 ? [[group, values] as [MetricGroup, Record<string, number>]] : [];
  });
}

/**
 * Dev-console panel: read-only view of offline evaluation results (BEIR / RAGAS
 * / retrieval / Bamboogle summaries) from the configured results directory,
 * with retrieval metrics shown apart from generation metrics.
 */
export function EvalResultsPanel() {
  const [results, setResults] = useState<EvalResultFile[] | null>(null);

  useEffect(() => {
    let alive = true;
    getEvalResults().then(
      (r) => alive && setResults(r.results),
      () => alive && setResults([]),
    );
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section className="eval-results" aria-label="Evaluation results">
      <h2>Evaluation Results</h2>
      {results !== null && results.length === 0 && (
        <p className="eval-results__empty">
          No eval results yet — run an eval with <code>--output</code> into{" "}
          <code>data/eval/</code>.
        </p>
      )}
      {results?.map((file) => {
        const groups = groupsOf(file);
        return (
          <article key={file.name} className="eval-results__card">
            <header>
              <span className="eval-results__name">{file.name}</span>
              <span className="eval-results__mtime">
                {new Date(file.modified * 1000).toISOString().slice(0, 19).replace("T", " ")}
              </span>
            </header>
            {groups.length === 0 ? (
              <p className="eval-results__empty">no numeric metrics</p>
            ) : (
              groups.map(([group, values]) => (
                <table key={group} className={`eval-results__group eval-results__group--${group}`}>
                  <caption>{GROUP_LABELS[group]}</caption>
                  <tbody>
                    {Object.entries(values).map(([k, v]) => (
                      <tr key={k}>
                        <td>{k}</td>
                        <td>{fmt(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ))
            )}
          </article>
        );
      })}
    </section>
  );
}
