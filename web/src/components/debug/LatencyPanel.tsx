import { useEffect, useState } from "react";
import { getRouteLatency } from "../../api";
import type { RouteLatencyRow, StageLatency } from "../../types";

/**
 * Milliseconds, or a dash.
 *
 * A finite-guard here rather than a bare `toFixed`: #437 shipped a crash from a
 * non-finite metric crossing FastAPI -> JS as null and reaching toFixed.
 */
function ms(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(1) : "—";
}

function num(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function pct(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(0)}%` : "—";
}

function StageTable({ retrieval, generation }: { retrieval: StageLatency; generation: StageLatency }) {
  return (
    <table className="latency-panel__table latency-panel__stages" aria-label="Stage latency">
      <caption>Retrieval vs generation (recent requests that used each stage)</caption>
      <thead>
        <tr>
          <th scope="col">Stage</th>
          <th scope="col">Requests</th>
          <th scope="col">p50 ms</th>
          <th scope="col">p95 ms</th>
          <th scope="col">max ms</th>
          <th scope="col">Detail</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">Retrieval</th>
          <td>{retrieval.count}</td>
          <td>{ms(retrieval.p50_ms)}</td>
          <td>{ms(retrieval.p95_ms)}</td>
          <td>{ms(retrieval.max_ms)}</td>
          <td>
            {num(retrieval.avg_docs)} docs · {pct(retrieval.cache_hit_rate)} cache hits
          </td>
        </tr>
        <tr>
          <th scope="row">Generation</th>
          <td>{generation.count}</td>
          <td>{ms(generation.p50_ms)}</td>
          <td>{ms(generation.p95_ms)}</td>
          <td>{ms(generation.max_ms)}</td>
          <td>
            {num(generation.avg_prompt_tokens, 0)} prompt · {num(generation.avg_completion_tokens, 0)}{" "}
            completion tokens
          </td>
        </tr>
      </tbody>
    </table>
  );
}

/**
 * Dev-console panel: per-route request latency, slowest p95 first, and the
 * same window split into its retrieval and generation shares.
 *
 * The Request Inspector shows where one request spent its time. This shows
 * which route to inspect, and which side of the pipeline to blame.
 */
export function LatencyPanel() {
  const [routes, setRoutes] = useState<RouteLatencyRow[] | null>(null);
  const [stages, setStages] = useState<{ retrieval: StageLatency; generation: StageLatency } | null>(null);

  useEffect(() => {
    let alive = true;
    getRouteLatency().then(
      (r) => {
        if (!alive) return;
        setRoutes(r.routes);
        setStages(r.stages ?? null);
      },
      () => alive && setRoutes([]),
    );
    return () => {
      alive = false;
    };
  }, []);

  return (
    <section className="latency-panel" aria-label="Route latency">
      <h2>Route Latency</h2>
      {routes !== null && routes.length === 0 && (
        <p className="latency-panel__empty">
          No requests recorded yet — issue a request, then reload this panel.
        </p>
      )}
      {routes !== null && routes.length > 0 && (
        <table className="latency-panel__table">
          <thead>
            <tr>
              <th scope="col">Route</th>
              <th scope="col">Calls</th>
              <th scope="col">Errors</th>
              <th scope="col">p50 ms</th>
              <th scope="col">p95 ms</th>
              <th scope="col">max ms</th>
            </tr>
          </thead>
          <tbody>
            {routes.map((row) => (
              <tr key={`${row.method} ${row.route}`}>
                <th scope="row">
                  <code>
                    {row.method} {row.route}
                  </code>
                </th>
                <td>{row.count}</td>
                <td className={row.errors > 0 ? "latency-panel__errors" : undefined}>
                  {row.errors}
                </td>
                <td>{ms(row.p50_ms)}</td>
                <td>{ms(row.p95_ms)}</td>
                <td>{ms(row.max_ms)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {stages && (stages.retrieval.count > 0 || stages.generation.count > 0) && (
        <StageTable retrieval={stages.retrieval} generation={stages.generation} />
      )}
    </section>
  );
}
