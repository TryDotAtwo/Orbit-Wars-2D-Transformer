import { useState, type PointerEvent } from 'react';
import type { MetricPoint, ModelRow } from './types';

const CHART_WIDTH = 360;
const CHART_HEIGHT = 118;
const CHART_PADDING = 24;
const CHART_POINT_RADIUS = 3;
const CHART_TOOLTIP_WIDTH = 116;
const CHART_TOOLTIP_HEIGHT = 26;
const CHART_TOOLTIP_TOP = 8;

type LineChartProps = {
  title: string;
  points: MetricPoint[];
  field: keyof Pick<
    MetricPoint,
    | 'winRate'
    | 'gamesPerSecond'
    | 'turnsPerSecond'
    | 'p95LatencyMs'
    | 'gpuUtilization'
    | 'avgModelActionMs'
    | 'generationSeconds'
    | 'captures'
    | 'fleetHits'
    | 'sunDestroyedFleets'
    | 'avgFleetSize'
  >;
  color: string;
  suffix: string;
};

export function LineChart({ title, points, field, color, suffix }: LineChartProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const chartPoints = visibleChartPoints(points, field);
  if (chartPoints.length === 0) {
    return null;
  }
  const values = chartPoints.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const latestPoint = chartPoints.at(-1);
  const latestValue = latestPoint?.value ?? 0;
  const hoveredPoint = hoveredIndex === null ? undefined : chartPoints[hoveredIndex];
  const hoveredValue = hoveredPoint?.value ?? 0;
  const hoveredX = hoveredIndex === null ? 0 : pointX(hoveredIndex, chartPoints.length);
  const hoveredY = pointY(hoveredValue, min, span);
  const tooltipX = Math.min(
    Math.max(CHART_PADDING, hoveredX - CHART_TOOLTIP_WIDTH / 2),
    CHART_WIDTH - CHART_PADDING - CHART_TOOLTIP_WIDTH,
  );
  const path = chartPoints
    .map((point, index) => {
      const x = pointX(index, chartPoints.length);
      const y = pointY(point.value, min, span);
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <section className="chart-panel" aria-label={title}>
      <div className="panel-header">
        <span>{title}</span>
        <strong>
          G{latestPoint?.metric.generation ?? 0}: {formatChartValue(latestValue, field)}
          {suffix}
        </strong>
      </div>
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        role="img"
        onPointerMove={(event) => setHoveredIndex(pointerIndex(event, chartPoints.length))}
        onPointerDown={(event) => setHoveredIndex(pointerIndex(event, chartPoints.length))}
        onPointerLeave={() => setHoveredIndex(null)}
      >
        <line
          x1={CHART_PADDING}
          y1={CHART_HEIGHT - CHART_PADDING}
          x2={CHART_WIDTH - CHART_PADDING}
          y2={CHART_HEIGHT - CHART_PADDING}
          className="axis"
        />
        <line
          x1={CHART_PADDING}
          y1={CHART_PADDING}
          x2={CHART_PADDING}
          y2={CHART_HEIGHT - CHART_PADDING}
          className="axis"
        />
        <path d={path} fill="none" stroke={color} strokeWidth="2.2" />
        {chartPoints.map((point, index) => {
          const x = pointX(index, chartPoints.length);
          const y = pointY(point.value, min, span);
          return <circle key={point.metric.generation} cx={x} cy={y} r={CHART_POINT_RADIUS} fill={color} />;
        })}
        {hoveredPoint ? (
          <g className="chart-hover">
            <line x1={hoveredX} y1={CHART_PADDING} x2={hoveredX} y2={CHART_HEIGHT - CHART_PADDING} />
            <circle cx={hoveredX} cy={hoveredY} r={CHART_POINT_RADIUS + 2} fill={color} />
            <rect x={tooltipX} y={CHART_TOOLTIP_TOP} width={CHART_TOOLTIP_WIDTH} height={CHART_TOOLTIP_HEIGHT} rx="5" />
            <text x={tooltipX + 8} y={CHART_TOOLTIP_TOP + 17}>
              G{hoveredPoint.metric.generation} {formatChartValue(hoveredValue, field)}
              {suffix}
            </text>
          </g>
        ) : null}
        <rect
          className="chart-hit-area"
          x="0"
          y="0"
          width={CHART_WIDTH}
          height={CHART_HEIGHT}
          aria-label={`${title} hover area`}
        />
      </svg>
    </section>
  );
}

function visibleChartPoints(points: MetricPoint[], field: LineChartProps['field']) {
  return points
    .map((metric) => ({ metric, value: Number(metric[field]) }))
    .filter((point) => Number.isFinite(point.value) && point.value !== 0);
}

function pointX(index: number, pointCount: number) {
  return CHART_PADDING + (index / Math.max(pointCount - 1, 1)) * (CHART_WIDTH - CHART_PADDING * 2);
}

function pointY(value: number, min: number, span: number) {
  return CHART_HEIGHT - CHART_PADDING - ((value - min) / span) * (CHART_HEIGHT - CHART_PADDING * 2);
}

function pointerIndex(event: PointerEvent<SVGSVGElement>, pointCount: number) {
  if (pointCount <= 1) {
    return 0;
  }
  const rect = event.currentTarget.getBoundingClientRect();
  const normalized = (event.clientX - rect.left) / Math.max(rect.width, 1);
  const chartX = normalized * CHART_WIDTH;
  const chartSpan = CHART_WIDTH - CHART_PADDING * 2;
  const chartRatio = (chartX - CHART_PADDING) / chartSpan;
  const boundedRatio = Math.min(Math.max(chartRatio, 0), 1);
  return Math.round(boundedRatio * (pointCount - 1));
}

function formatChartValue(value: number, field: LineChartProps['field']) {
  if (field === 'winRate' || field === 'avgModelActionMs' || field === 'avgFleetSize') {
    return value.toFixed(2);
  }
  if (field === 'generationSeconds') {
    return value.toFixed(1);
  }
  return value.toFixed(0);
}

export function ModelTable({ models }: { models: ModelRow[] }) {
  return (
    <section className="model-table" aria-label="model pool">
      <div className="panel-header">
        <span>Model pool</span>
        <strong>top 12 gate</strong>
      </div>
      <table>
        <thead>
          <tr>
            <th>model</th>
            <th>parent</th>
            <th>rating</th>
            <th>record</th>
            <th>captures</th>
            <th>hits</th>
            <th>sun</th>
            <th>avg fleet</th>
            <th>ships sent</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model) => (
            <tr key={model.id} className={model.selected ? 'selected-row' : undefined}>
              <td>{model.id}</td>
              <td>{model.parent}</td>
              <td>{model.rating}</td>
              <td>
                {model.wins}/{model.draws}/{model.losses}
              </td>
              <td>{model.captures}</td>
              <td>
                {model.fleetHits}/{model.hitShips}
              </td>
              <td>
                {model.sunDestroyedFleets}/{model.sunDestroyedShips}
              </td>
              <td>{model.avgFleetSize.toFixed(1)}</td>
              <td>{model.launchedShips}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
