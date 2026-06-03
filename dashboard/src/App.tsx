import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { LineChart, ModelTable } from './Charts';
import { ReplayCanvas } from './ReplayCanvas';
import { loadReplayChunk, loadReplayGame, loadTelemetry, sampleTelemetry } from './sampleData';
import type { DashboardTelemetry, GenerationWinRatePoint, MetricPoint, ModelRow, ReplayGame } from './types';

const TAB_LIVE_TRAINING = 'Live Training';
const TAB_GENERATION_LOGS = 'Generation Logs';
const TAB_REPLAY = 'Replay';
const TAB_GENERATION_WINRATE = 'Generation Winrate';
const TAB_MODELS = 'Models';
const TAB_SUBMISSIONS = 'Submissions';
const TAB_ARTIFACTS = 'Artifacts';
const TABS = [
  TAB_LIVE_TRAINING,
  TAB_GENERATION_LOGS,
  TAB_REPLAY,
  TAB_GENERATION_WINRATE,
  TAB_MODELS,
  TAB_SUBMISSIONS,
  TAB_ARTIFACTS,
];
const TELEMETRY_POLL_INTERVAL_MS = 2000;
const REPLAY_PLAY_INTERVAL_MS = 120;
const ALL_GENERATIONS_FILTER = 'all';
const ALL_MODELS_FILTER = 'all';
const STATUS_NUMBER_LOCALE = 'en-US';
const WIN_RATE_PERCENT_MULTIPLIER = 100;
const GENERATION_WINRATE_RECENT_VALIDATION_COUNT = 8;
const GENERATION_MATRIX_LABEL_COLUMN_WIDTH_PX = 88;
const GENERATION_MATRIX_DATA_COLUMN_MIN_WIDTH_PX = 58;
const GENERATION_MATRIX_ALPHA_BASE = 0.08;
const GENERATION_MATRIX_ALPHA_RANGE = 0.32;
const IN_PROGRESS_STATUS_MARKER = 'status=in_progress';

export function App() {
  const [telemetry, setTelemetry] = useState<DashboardTelemetry>(sampleTelemetry);
  const [activeTab, setActiveTab] = useState(TABS[0]);
  const [selectedGeneration, setSelectedGeneration] = useState(ALL_GENERATIONS_FILTER);
  const [selectedModelId, setSelectedModelId] = useState(ALL_MODELS_FILTER);
  const [selectedReplayIndex, setSelectedReplayIndex] = useState(0);
  const [selectedFrameIndex, setSelectedFrameIndex] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [chunkReplayGames, setChunkReplayGames] = useState<ReplayGame[]>([]);
  const loadedReplayChunkPaths = useRef<Set<string>>(new Set());
  const loadedReplayGamePaths = useRef<Set<string>>(new Set());
  const replayGames = useMemo(
    () => mergeReplayGames(chunkReplayGames, telemetry.replayGames),
    [chunkReplayGames, telemetry.replayGames],
  );
  const generationOptions = useMemo(
    () =>
      Array.from(new Set([...replayGames.map((game) => game.generation), ...telemetry.replayChunks.map((chunk) => chunk.generation)])).sort(
        (left, right) => right - left,
      ),
    [replayGames, telemetry.replayChunks],
  );
  const replayChunkPathKey = useMemo(
    () => telemetry.replayChunks.map((chunk) => chunk.path).join('|'),
    [telemetry.replayChunks],
  );
  const modelOptions = useMemo(
    () =>
      Array.from(
        new Set(
          replayGames
            .filter((game) => selectedGeneration === ALL_GENERATIONS_FILTER || game.generation === Number(selectedGeneration))
            .map((game) => game.modelId),
        ),
      ).sort(),
    [replayGames, selectedGeneration],
  );
  const filteredReplayGames = useMemo(
    () =>
      replayGames.filter(
        (game) =>
          (selectedGeneration === ALL_GENERATIONS_FILTER || game.generation === Number(selectedGeneration)) &&
          (selectedModelId === ALL_MODELS_FILTER || game.modelId === selectedModelId),
      ),
    [replayGames, selectedGeneration, selectedModelId],
  );
  const selectableReplayGames = useMemo(
    () =>
      selectedModelId === ALL_MODELS_FILTER
        ? actualReplayGames(filteredReplayGames)
        : filteredReplayGames,
    [filteredReplayGames, selectedModelId],
  );
  const boundedReplayIndex = Math.min(selectedReplayIndex, Math.max(0, selectableReplayGames.length - 1));
  const selectedReplay = selectableReplayGames[boundedReplayIndex];
  const usingLiveReplayFeed =
    selectedReplay !== undefined &&
    telemetry.replayGames.some((game) => replayGameKey(game) === replayGameKey(selectedReplay));
  const loadedReplayFrames = selectedReplay?.frames.length
    ? selectedReplay.frames
    : selectedGeneration === ALL_GENERATIONS_FILTER || selectedGeneration === String(telemetry.activeGeneration)
      ? telemetry.frames
      : [];
  const replayFrames = loadedReplayFrames.length ? loadedReplayFrames : sampleTelemetry.frames;
  const boundedFrameIndex = Math.min(selectedFrameIndex, Math.max(0, replayFrames.length - 1));
  const frame = replayFrames[boundedFrameIndex] ?? sampleTelemetry.frames[0];
  const latest = telemetry.metrics[telemetry.metrics.length - 1] ?? sampleTelemetry.metrics[sampleTelemetry.metrics.length - 1];
  const fullGameFrameStatus =
    telemetry.fullReplay && replayFrames.length === telemetry.expectedFramesPerGame ? 'complete' : 'partial';
  const selectedModel = useMemo(
    () => telemetry.models.find((model) => model.selected) ?? telemetry.models[0],
    [telemetry.models],
  );

  useEffect(() => {
    let mounted = true;
    const refreshTelemetry = () => {
      loadTelemetry({ includeLiveReplay: activeTab === TAB_REPLAY }).then((loadedTelemetry) => {
        if (mounted) {
          setTelemetry(loadedTelemetry);
        }
      });
    };
    refreshTelemetry();
    const interval = window.setInterval(refreshTelemetry, TELEMETRY_POLL_INTERVAL_MS);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, [activeTab]);

  useEffect(() => {
    setSelectedReplayIndex(0);
    setSelectedFrameIndex(0);
    setReplayPlaying(false);
  }, [selectedGeneration, selectedModelId]);

  useEffect(() => {
    setChunkReplayGames([]);
    loadedReplayChunkPaths.current.clear();
    loadedReplayGamePaths.current.clear();
    setReplayPlaying(false);
    setSelectedGeneration(String(telemetry.activeGeneration));
    setSelectedModelId(ALL_MODELS_FILTER);
  }, [telemetry.runId]);

  useEffect(() => {
    if (activeTab !== TAB_REPLAY || selectedGeneration === ALL_GENERATIONS_FILTER) {
      return;
    }
    const pendingChunks = telemetry.replayChunks.filter(
      (chunk) =>
        !loadedReplayChunkPaths.current.has(chunk.path) &&
        chunk.generation === Number(selectedGeneration),
    );
    if (pendingChunks.length === 0) {
      return;
    }
    const pendingPaths = pendingChunks.map((chunk) => chunk.path);
    let cancelled = false;
    Promise.all(pendingChunks.map((chunk) => loadReplayChunk(chunk.path)))
      .then((chunks) => {
        if (cancelled) {
          return;
        }
        pendingPaths.forEach((path) => loadedReplayChunkPaths.current.add(path));
        setChunkReplayGames((previous) => mergeReplayGames(previous, chunks.flat()));
      })
      .catch(() => {
        if (!cancelled) {
          pendingPaths.forEach((path) => loadedReplayChunkPaths.current.delete(path));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activeTab, replayChunkPathKey, selectedGeneration]);

  useEffect(() => {
    if (activeTab !== TAB_REPLAY || !selectedReplay?.gamePath || loadedReplayGamePaths.current.has(selectedReplay.gamePath)) {
      return;
    }
    const gamePath = selectedReplay.gamePath;
    let cancelled = false;
    loadReplayGame(gamePath)
      .then((games) => {
        if (cancelled) {
          return;
        }
        loadedReplayGamePaths.current.add(gamePath);
        setChunkReplayGames((previous) => mergeReplayGames(previous, games));
      })
      .catch(() => {
        loadedReplayGamePaths.current.delete(gamePath);
      });
    return () => {
      cancelled = true;
    };
  }, [activeTab, selectedReplay?.gamePath]);

  useEffect(() => {
    setSelectedFrameIndex(0);
    setReplayPlaying(false);
  }, [selectedReplayIndex]);

  useEffect(() => {
    if (!replayPlaying || replayFrames.length <= 1) {
      return;
    }
    const interval = window.setInterval(() => {
      setSelectedFrameIndex((currentFrameIndex) => {
        const lastFrameIndex = Math.max(0, replayFrames.length - 1);
        return currentFrameIndex >= lastFrameIndex ? 0 : currentFrameIndex + 1;
      });
    }, REPLAY_PLAY_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [replayFrames.length, replayPlaying]);

  useEffect(() => {
    if (!usingLiveReplayFeed || replayPlaying) {
      return;
    }
    setSelectedFrameIndex(Math.max(0, replayFrames.length - 1));
  }, [replayFrames.length, replayPlaying, usingLiveReplayFeed]);

  return (
    <main className="app-shell">
      <aside className="left-rail" aria-label="navigation">
        <div className="brand-mark">OW</div>
        {TABS.map((tab) => (
          <button
            key={tab}
            className={activeTab === tab ? 'nav-button active' : 'nav-button'}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </aside>

      <section className="workspace">
        <header className="top-bar">
          <div>
            <h1>Orbit Wars Trainer</h1>
            <p>{telemetry.sourceMessage}</p>
          </div>
          <Status label="generation" value={String(telemetry.activeGeneration)} tone="green" />
          <Status label="turns/sec" value={formatNumber(telemetry.turnsPerSecond)} tone="cyan" />
          <Status label="step limit" value={String(telemetry.episodeSteps)} tone="amber" />
          <Status label="action avg" value={`${latest.avgModelActionMs.toFixed(3)} ms`} tone="red" />
        </header>

        <div className="content-grid">
          <section className="main-column">
            {activeTab === TAB_LIVE_TRAINING ? (
              <LiveTrainingView telemetry={telemetry} latest={latest} />
            ) : null}
            {activeTab === TAB_GENERATION_LOGS ? (
              <GenerationLogsView
                telemetry={telemetry}
                onOpenReplay={(generation) => {
                  setSelectedGeneration(String(generation));
                  setActiveTab(TAB_REPLAY);
                }}
              />
            ) : null}
            {activeTab === TAB_REPLAY ? (
              <>
                <ReplayControls
                  generationOptions={generationOptions}
                  modelOptions={modelOptions}
                  replayGames={selectableReplayGames}
                  telemetry={telemetry}
                  selectedGeneration={selectedGeneration}
                  selectedModelId={selectedModelId}
                  selectedReplayIndex={boundedReplayIndex}
                  selectedReplay={selectedReplay}
                  replayFrameCount={replayFrames.length}
                  fullGameFrameStatus={fullGameFrameStatus}
                  onGeneration={setSelectedGeneration}
                  onModel={setSelectedModelId}
                  onReplay={setSelectedReplayIndex}
                />
                <ReplayCanvas
                  frame={frame}
                  frames={replayFrames}
                  frameIndex={boundedFrameIndex}
                  frameCount={replayFrames.length}
                  isPlaying={replayPlaying}
                  onFrameIndex={setSelectedFrameIndex}
                  onTogglePlaying={() => setReplayPlaying((current) => !current)}
                />
                <ReplayFactGrid telemetry={telemetry} frameCount={replayFrames.length} selectedReplay={selectedReplay} />
              </>
            ) : null}
            {activeTab === TAB_GENERATION_WINRATE ? <GenerationWinrateView telemetry={telemetry} /> : null}
            {activeTab === TAB_MODELS ? <ModelTable models={telemetry.models} /> : null}
            {activeTab === TAB_SUBMISSIONS ? <SubmissionPanel telemetry={telemetry} /> : null}
            {activeTab === TAB_ARTIFACTS ? <ArtifactPanel telemetry={telemetry} /> : null}
          </section>

          <aside className="inspector">
            <RunFacts telemetry={telemetry} latest={latest} />
            <SelectedModelCard selectedModel={selectedModel} telemetry={telemetry} />
            <WarningsPanel telemetry={telemetry} />
          </aside>
        </div>
        <footer className="footer-line">
          active_view={activeTab}; latest_generation={latest?.generation ?? telemetry.activeGeneration};
          data_contract=append_only_artifacts
        </footer>
      </section>
    </main>
  );
}

function LiveTrainingView({ telemetry, latest }: { telemetry: DashboardTelemetry; latest: MetricPoint }) {
  const metricTiles = [
    { label: 'evaluated games', value: String(latest.evaluatedGames), detail: 'selection batch', tone: 'green', metricValue: latest.evaluatedGames },
    { label: 'replay games', value: String(latest.sampledReplayGames), detail: 'dashboard samples', tone: 'cyan', metricValue: latest.sampledReplayGames },
    { label: 'model calls', value: formatNumber(latest.modelActionCalls), detail: 'encode-forward-decode', tone: 'amber', metricValue: latest.modelActionCalls },
    {
      label: 'launch actions',
      value: formatNumber(latest.launchActions),
      detail: `${latest.avgLaunchActionsPerTurn.toFixed(2)} per turn`,
      tone: 'green',
      metricValue: latest.launchActions,
    },
    {
      label: 'ships launched',
      value: formatNumber(latest.launchedShips),
      detail: `${latest.avgLaunchedShipsPerTurn.toFixed(1)} ships per turn`,
      tone: 'cyan',
      metricValue: latest.launchedShips,
    },
    { label: 'captures', value: formatNumber(latest.captures), detail: `${formatNumber(latest.fleetHits)} fleet hits`, tone: 'amber', metricValue: latest.captures },
    {
      label: 'sun losses',
      value: formatNumber(latest.sunDestroyedFleets),
      detail: `${formatNumber(latest.sunDestroyedShips)} ships burned`,
      tone: 'red',
      metricValue: latest.sunDestroyedFleets,
    },
    { label: 'avg fleet', value: latest.avgFleetSize.toFixed(1), detail: `${formatNumber(latest.hitShips)} hit ships`, tone: 'green', metricValue: latest.avgFleetSize },
    {
      label: 'batch max',
      value: formatNumber(latest.maxInferenceBatchSize),
      detail: `${formatNumber(latest.simultaneousGames)} games together`,
      tone: 'cyan',
      metricValue: latest.maxInferenceBatchSize,
    },
    {
      label: 'backprop samples',
      value: formatNumber(latest.backpropSamples),
      detail: `${latest.backpropModels} trained models`,
      tone: 'amber',
      metricValue: latest.backpropSamples,
    },
  ];
  return (
    <>
      <section className="metric-strip" aria-label="current training metrics">
        {metricTiles
          .filter((tile) => metricValueIsVisible(tile.metricValue))
          .map((tile) => (
            <MetricTile key={tile.label} label={tile.label} value={tile.value} detail={tile.detail} tone={tile.tone} />
          ))}
      </section>
      <div className="chart-grid">
        <LineChart title="Win rate" points={telemetry.metrics} field="winRate" color="#55d986" suffix="" />
        <LineChart title="Games/s" points={telemetry.metrics} field="gamesPerSecond" color="#44b8ff" suffix="/s" />
        <LineChart title="Turns/s" points={telemetry.metrics} field="turnsPerSecond" color="#7dd3fc" suffix="/s" />
        <LineChart title="Action avg" points={telemetry.metrics} field="avgModelActionMs" color="#f5c542" suffix=" ms" />
        <LineChart title="Captures" points={telemetry.metrics} field="captures" color="#ff8f57" suffix="" />
        <LineChart title="Sun losses" points={telemetry.metrics} field="sunDestroyedFleets" color="#ff5f57" suffix="" />
        <LineChart title="Fleet hits" points={telemetry.metrics} field="fleetHits" color="#8edbff" suffix="" />
        <LineChart title="Avg fleet" points={telemetry.metrics} field="avgFleetSize" color="#d7b8ff" suffix="" />
        <LineChart title="Generation" points={telemetry.metrics} field="generationSeconds" color="#ff8f57" suffix=" s" />
      </div>
      <ComputeBreakdown latest={latest} />
    </>
  );
}

function MetricTile({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  tone: string;
}) {
  return (
    <div className={`metric-tile ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function metricValueIsVisible(value: number) {
  return Number.isFinite(value) && value !== 0;
}

function ComputeBreakdown({ latest }: { latest: MetricPoint }) {
  const rows = [
    { label: 'generation', value: latest.generationSeconds, color: '#55d986' },
    { label: 'evaluation', value: latest.evaluationSeconds, color: '#44b8ff' },
    { label: 'backprop', value: latest.backpropSeconds, color: '#b7e36f' },
    { label: 'generation tournament', value: latest.generationValidationSeconds, color: '#d7b8ff' },
    { label: 'model action', value: latest.modelActionSeconds, color: '#ff8f57' },
    { label: 'simulator step', value: latest.simulationStepSeconds, color: '#ff5f57' },
    { label: 'replay write', value: latest.replayWriteSeconds, color: '#8edbff' },
    { label: 'reproduction', value: latest.reproductionSeconds, color: '#f5c542' },
  ];
  const visibleRows = rows.filter((row) => metricValueIsVisible(row.value));
  if (visibleRows.length === 0) {
    return null;
  }
  const maxValue = Math.max(...visibleRows.map((row) => row.value), 1);
  return (
    <section className="breakdown-panel" aria-label="compute time breakdown">
      <div className="panel-header">
        <span>Compute time</span>
        <strong>generation {latest.generation}</strong>
      </div>
      <div className="breakdown-list">
        {visibleRows.map((row) => (
          <div className="breakdown-row" key={row.label}>
            <span>{row.label}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(row.value / maxValue) * 100}%`, background: row.color }} />
            </div>
            <strong>{row.value.toFixed(3)} s</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function GenerationLogsView({
  telemetry,
  onOpenReplay,
}: {
  telemetry: DashboardTelemetry;
  onOpenReplay: (generation: number) => void;
}) {
  const rows = generationLogRows(telemetry);
  const archivedActualGames = rows.reduce((sum, row) => sum + row.actualGameCount, 0);
  const archivedViews = rows.reduce((sum, row) => sum + row.replayViewCount, 0);
  return (
    <>
      <section className="metric-strip" aria-label="generation log facts">
        <MetricTile label="logged generations" value={String(rows.length)} detail="metric rows" tone="green" />
        <MetricTile label="replay chunks" value={String(telemetry.replayChunks.length)} detail="generation files" tone="cyan" />
        <MetricTile label="archived games" value={String(archivedActualGames)} detail={`${telemetry.generationReplayGameCount} target per generation`} tone="amber" />
        <MetricTile label="replay views" value={String(archivedViews)} detail={`${telemetry.playersPerGame} player views per game`} tone="green" />
        <MetricTile label="live frames" value={String(telemetry.frames.length)} detail="latest telemetry" tone="cyan" />
      </section>
      <section className="wide-panel generation-log-panel" aria-label="generation logs">
        <div className="panel-header">
          <span>Generation logs</span>
          <strong>run {telemetry.runId}</strong>
        </div>
        <div className="generation-log-table-wrap">
          <table className="generation-log-table">
            <thead>
              <tr>
                <th>generation</th>
                <th>status</th>
                <th>eval games</th>
                <th>replay games</th>
                <th>views</th>
                <th>turns/s</th>
                <th>action avg</th>
                <th>eval</th>
                <th>backprop</th>
                <th>write</th>
                <th>total</th>
                <th>batch</th>
                <th>open</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`generation-log-${row.metric.generation}`}>
                  <td>G{row.metric.generation}</td>
                  <td>
                    <span className={`status-pill ${row.status}`}>{row.status}</span>
                  </td>
                  <td>{formatNumber(row.metric.evaluatedGames)}</td>
                  <td>{row.actualGameCount}</td>
                  <td>{row.replayViewCount}</td>
                  <td>{formatNumber(row.metric.turnsPerSecond)}</td>
                  <td>{row.metric.avgModelActionMs.toFixed(3)} ms</td>
                  <td>{formatSeconds(row.metric.evaluationSeconds)}</td>
                  <td>{formatSeconds(row.metric.backpropSeconds)}</td>
                  <td>{formatSeconds(row.metric.replayWriteSeconds)}</td>
                  <td>{formatSeconds(row.metric.generationSeconds)}</td>
                  <td>{formatNumber(row.metric.maxInferenceBatchSize)}</td>
                  <td>
                    <button className="inline-action" type="button" onClick={() => onOpenReplay(row.metric.generation)}>
                      replay
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

function generationLogRows(telemetry: DashboardTelemetry) {
  const metricsByGeneration = new Map<number, MetricPoint>();
  for (const metric of telemetry.metrics) {
    metricsByGeneration.set(metric.generation, metric);
  }
  const replayChunksByGeneration = new Map(telemetry.replayChunks.map((chunk) => [chunk.generation, chunk]));
  return Array.from(metricsByGeneration.values())
    .sort((left, right) => right.generation - left.generation)
    .map((metric) => {
      const replayChunk = replayChunksByGeneration.get(metric.generation);
      return {
        metric,
        replayViewCount: replayChunk?.gameCount ?? 0,
        actualGameCount: replayChunk ? replayActualGameCount(replayChunk, telemetry.playersPerGame) : 0,
        status: generationLogStatus(metric, telemetry),
      };
    });
}

function generationLogStatus(metric: MetricPoint, telemetry: DashboardTelemetry) {
  if (metric.generation === telemetry.activeGeneration && telemetry.sourceMessage.includes(IN_PROGRESS_STATUS_MARKER)) {
    return 'running';
  }
  if (metric.generationSeconds > 0) {
    return 'complete';
  }
  return 'pending';
}

function replayActualGameCount(chunk: DashboardTelemetry['replayChunks'][number], playersPerGame: number) {
  return chunk.actualGameCount ?? Math.ceil(chunk.gameCount / Math.max(playersPerGame, 1));
}


function GenerationWinrateView({ telemetry }: { telemetry: DashboardTelemetry }) {
  const points = telemetry.generationWinRates;
  const validationGenerations = uniqueDescending(points.map((point) => point.validationGeneration)).slice(
    0,
    GENERATION_WINRATE_RECENT_VALIDATION_COUNT,
  );
  const evaluatedGenerations = uniqueAscending(points.map((point) => point.evaluatedGeneration));
  const latestValidationGeneration = validationGenerations[0] ?? telemetry.activeGeneration;
  const latestRows = points
    .filter((point) => point.validationGeneration === latestValidationGeneration)
    .sort((left, right) => right.winRate - left.winRate || right.evaluatedGeneration - left.evaluatedGeneration);
  const pointByKey = new Map(points.map((point) => [generationWinRateKey(point.validationGeneration, point.evaluatedGeneration), point]));
  const latestChampionModels = latestRows.reduce((sum, point) => sum + point.modelCount, 0);
  const latestValidationParticipations = latestRows.reduce((sum, point) => sum + point.games, 0);
  const latestTournamentGames =
    telemetry.metrics.find((metric) => metric.generation === latestValidationGeneration)?.generationValidationGames ??
    Math.ceil(latestValidationParticipations / Math.max(telemetry.playersPerGame, 1));
  const bestGeneration = latestRows[0]?.evaluatedGeneration ?? 0;
  return (
    <>
      <section className="metric-strip" aria-label="generation tournament facts">
        <MetricTile label="tournament stage" value={`G${latestValidationGeneration}`} detail="latest generation check" tone="green" />
        <MetricTile label="tracked generations" value={String(latestRows.length)} detail="max 32 generations" tone="cyan" />
        <MetricTile label="champion models" value={String(latestChampionModels)} detail="max 128 models" tone="amber" />
        <MetricTile label="tournament games" value={String(latestTournamentGames)} detail="4-player matches" tone="green" />
        <MetricTile label="best generation" value={`G${bestGeneration}`} detail="highest latest win rate" tone="cyan" />
      </section>
      <section className="wide-panel generation-winrate-panel" aria-label="generation win rate matrix">
        <div className="panel-header">
          <span>Generation win rate</span>
          <strong>rows=tournament stage; columns=evaluated generation</strong>
        </div>
        {points.length === 0 ? (
          <p className="fine-print">generation_tournament_records=empty</p>
        ) : (
          <div
            className="generation-matrix"
            style={{
              gridTemplateColumns: `${GENERATION_MATRIX_LABEL_COLUMN_WIDTH_PX}px repeat(${Math.max(
                evaluatedGenerations.length,
                1,
              )}, minmax(${GENERATION_MATRIX_DATA_COLUMN_MIN_WIDTH_PX}px, 1fr))`,
            }}
          >
            <div className="generation-matrix-corner">check</div>
            {evaluatedGenerations.map((generation) => (
              <div className="generation-matrix-head" key={`head-${generation}`}>
                G{generation}
              </div>
            ))}
            {validationGenerations.map((validationGeneration) => (
              <Fragment key={`validation-${validationGeneration}`}>
                <div className="generation-matrix-row-head">G{validationGeneration}</div>
                {evaluatedGenerations.map((evaluatedGeneration) => {
                  const point = pointByKey.get(generationWinRateKey(validationGeneration, evaluatedGeneration));
                  return (
                    <div
                      className="generation-matrix-cell"
                      key={`${validationGeneration}-${evaluatedGeneration}`}
                      title={generationWinRateTitle(point, validationGeneration, evaluatedGeneration)}
                      style={{ background: generationWinRateBackground(point?.winRate ?? 0) }}
                    >
                      {point ? formatWinRatePercent(point.winRate) : '-'}
                    </div>
                  );
                })}
              </Fragment>
            ))}
          </div>
        )}
      </section>
      <section className="model-table" aria-label="latest generation tournament ranking">
        <div className="panel-header">
          <span>Latest tournament ranking</span>
          <strong>tournament G{latestValidationGeneration}</strong>
        </div>
        <table>
          <thead>
            <tr>
              <th>generation</th>
              <th>models</th>
              <th>win rate</th>
              <th>record</th>
              <th>participations</th>
            </tr>
          </thead>
          <tbody>
            {latestRows.map((point) => (
              <tr key={`latest-${point.evaluatedGeneration}`}>
                <td>G{point.evaluatedGeneration}</td>
                <td>{point.modelCount}</td>
                <td>{formatWinRatePercent(point.winRate)}</td>
                <td>
                  {point.wins}/{point.draws}/{point.losses}
                </td>
                <td>{point.games}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}

function uniqueDescending(values: number[]) {
  return Array.from(new Set(values)).sort((left, right) => right - left);
}

function uniqueAscending(values: number[]) {
  return Array.from(new Set(values)).sort((left, right) => left - right);
}

function generationWinRateKey(validationGeneration: number, evaluatedGeneration: number) {
  return `${validationGeneration}:${evaluatedGeneration}`;
}

function formatWinRatePercent(winRate: number) {
  return `${Math.round(winRate * WIN_RATE_PERCENT_MULTIPLIER)}%`;
}

function generationWinRateBackground(winRate: number) {
  const alpha = GENERATION_MATRIX_ALPHA_BASE + winRate * GENERATION_MATRIX_ALPHA_RANGE;
  return `rgba(85, 217, 134, ${alpha.toFixed(3)})`;
}

function generationWinRateTitle(
  point: GenerationWinRatePoint | undefined,
  validationGeneration: number,
  evaluatedGeneration: number,
) {
  if (!point) {
    return `validation_generation=${validationGeneration}; evaluated_generation=${evaluatedGeneration}; status=missing`;
  }
  return `validation_generation=${point.validationGeneration}; evaluated_generation=${point.evaluatedGeneration}; win_rate=${formatWinRatePercent(point.winRate)}; record=${point.wins}/${point.draws}/${point.losses}; models=${point.modelCount}; participations=${point.games}`;
}

function ReplayFactGrid({
  telemetry,
  frameCount,
  selectedReplay,
}: {
  telemetry: DashboardTelemetry;
  frameCount: number;
  selectedReplay: ReplayGame | undefined;
}) {
  return (
    <section className="fact-grid" aria-label="replay facts">
      <MetricTile label="game reward" value={String(selectedReplay?.reward ?? 0)} detail="win=2 draw=-1 loss=-2" tone="green" />
      <MetricTile label="frames" value={`${frameCount}/${telemetry.expectedFramesPerGame}`} detail="step 0 plus 500 turns" tone="cyan" />
      <MetricTile label="stride" value={String(telemetry.replayFrameStride)} detail="stored turn interval" tone="amber" />
      <MetricTile label="game length" value={String(telemetry.episodeSteps)} detail="official turn cap" tone="red" />
    </section>
  );
}

function RunFacts({ telemetry, latest }: { telemetry: DashboardTelemetry; latest: MetricPoint }) {
  return (
    <section className="inspector-card">
      <div className="panel-header">
        <span>Run facts</span>
        <strong>{telemetry.source}</strong>
      </div>
      <dl>
        <div>
          <dt>profile</dt>
          <dd>{telemetry.runProfile}</dd>
        </div>
        <div>
          <dt>mode</dt>
          <dd>{telemetry.trainingMode}</dd>
        </div>
        <div>
          <dt>strict rules</dt>
          <dd>{String(telemetry.strictGameRules)}</dd>
        </div>
        <div>
          <dt>steps</dt>
          <dd>{telemetry.episodeSteps}</dd>
        </div>
        <div>
          <dt>population</dt>
          <dd>
            {telemetry.populationSize}/{telemetry.eliteCount}
          </dd>
        </div>
        <div>
          <dt>games/model</dt>
          <dd>{telemetry.gamesPerModel}</dd>
        </div>
        <div>
          <dt>players/game</dt>
          <dd>{telemetry.playersPerGame}</dd>
        </div>
        <div>
          <dt>sim games</dt>
          <dd>{telemetry.simultaneousGames}</dd>
        </div>
        <div>
          <dt>batch max</dt>
          <dd>{telemetry.maxInferenceBatchSize}</dd>
        </div>
        <div>
          <dt>tournament games</dt>
          <dd>{latest.generationValidationGames}</dd>
        </div>
        <div>
          <dt>replays/model</dt>
          <dd>{telemetry.replaysPerModel}</dd>
        </div>
        <div>
          <dt>gen replays</dt>
          <dd>{telemetry.generationReplayGameCount}</dd>
        </div>
        <div>
          <dt>generation sec</dt>
          <dd>{latest.generationSeconds.toFixed(2)}</dd>
        </div>
      </dl>
      <p className="fine-print">map_source={telemetry.mapSource}; turn_loop={telemetry.turnLoop}</p>
    </section>
  );
}

function SelectedModelCard({
  selectedModel,
  telemetry,
}: {
  selectedModel: ModelRow;
  telemetry: DashboardTelemetry;
}) {
  return (
    <section className="inspector-card">
      <div className="panel-header">
        <span>Selected model</span>
        <strong>{selectedModel.id}</strong>
      </div>
      <dl>
        <div>
          <dt>rating</dt>
          <dd>{selectedModel.rating}</dd>
        </div>
        <div>
          <dt>record</dt>
          <dd>
            {selectedModel.wins}/{selectedModel.draws}/{selectedModel.losses}
          </dd>
        </div>
        <div>
          <dt>captures</dt>
          <dd>{selectedModel.captures}</dd>
        </div>
        <div>
          <dt>hits</dt>
          <dd>
            {selectedModel.fleetHits}/{selectedModel.hitShips}
          </dd>
        </div>
        <div>
          <dt>sun</dt>
          <dd>
            {selectedModel.sunDestroyedFleets}/{selectedModel.sunDestroyedShips}
          </dd>
        </div>
        <div>
          <dt>avg fleet</dt>
          <dd>{selectedModel.avgFleetSize.toFixed(1)}</dd>
        </div>
        <div>
          <dt>mutation</dt>
          <dd>{selectedModel.mutation}</dd>
        </div>
        <div>
          <dt>workers</dt>
          <dd>{telemetry.cpuWorkers}</dd>
        </div>
      </dl>
    </section>
  );
}

function WarningsPanel({ telemetry }: { telemetry: DashboardTelemetry }) {
  const messages = [...telemetry.validationErrors, ...telemetry.warnings];
  return (
    <section className="inspector-card">
      <div className="panel-header">
        <span>Checks</span>
        <strong>{messages.length === 0 ? 'clear' : `${messages.length} item`}</strong>
      </div>
      {messages.length === 0 ? (
        <p className="ok-text">validation_errors=0; warnings=0</p>
      ) : (
        <ul className="error-list">
          {messages.map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      )}
    </section>
  );
}

function SubmissionPanel({ telemetry }: { telemetry: DashboardTelemetry }) {
  const latest = telemetry.metrics[telemetry.metrics.length - 1] ?? sampleTelemetry.metrics[0];
  return (
    <section className="wide-panel">
      <div className="panel-header">
        <span>Submission gate</span>
        <strong>{telemetry.trainingMode}</strong>
      </div>
      <p className="gate-text">{telemetry.submissionGate}</p>
      <div className="fact-grid compact">
        <MetricTile label="act timeout" value="1 s" detail="official limit tracked separately" tone="green" />
        <MetricTile label="episode" value={String(telemetry.episodeSteps)} detail="strict run turn cap" tone="cyan" />
        <MetricTile label="p95 game" value={`${latest.p95LatencyMs.toFixed(1)} ms`} detail="local evaluation" tone="amber" />
        <MetricTile label="kaggle submit" value="blocked" detail="user approval required" tone="red" />
      </div>
    </section>
  );
}

function ArtifactPanel({ telemetry }: { telemetry: DashboardTelemetry }) {
  return (
    <section className="wide-panel">
      <div className="panel-header">
        <span>Artifacts</span>
        <strong>{telemetry.runId}</strong>
      </div>
      <div className="artifact-list">
        <div>
          <span>latest telemetry</span>
          <strong>/telemetry/latest.json</strong>
        </div>
        {telemetry.replayChunks.map((chunk) => (
          <div key={chunk.path}>
            <span>G{chunk.generation} replay chunk</span>
            <strong>
              {replayActualGameCount(chunk, telemetry.playersPerGame)} games; {chunk.gameCount} views; {chunk.path}
            </strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function ReplayControls({
  generationOptions,
  modelOptions,
  replayGames,
  telemetry,
  selectedGeneration,
  selectedModelId,
  selectedReplayIndex,
  selectedReplay,
  replayFrameCount,
  fullGameFrameStatus,
  onGeneration,
  onModel,
  onReplay,
}: {
  generationOptions: number[];
  modelOptions: string[];
  replayGames: ReplayGame[];
  telemetry: DashboardTelemetry;
  selectedGeneration: string;
  selectedModelId: string;
  selectedReplayIndex: number;
  selectedReplay: ReplayGame | undefined;
  replayFrameCount: number;
  fullGameFrameStatus: string;
  onGeneration: (value: string) => void;
  onModel: (value: string) => void;
  onReplay: (value: number) => void;
}) {
  const selectsActualGames = selectedModelId === ALL_MODELS_FILTER;
  return (
    <section className="replay-controls" aria-label="replay selection">
      <label>
        <span>generation</span>
        <select value={selectedGeneration} onChange={(event) => onGeneration(event.currentTarget.value)}>
          <option value={ALL_GENERATIONS_FILTER}>all generations</option>
          {generationOptions.map((generation) => (
            <option key={generation} value={generation}>
              G{generation}
            </option>
          ))}
        </select>
      </label>
      <label>
        <span>model</span>
        <select value={selectedModelId} onChange={(event) => onModel(event.currentTarget.value)}>
          <option value={ALL_MODELS_FILTER}>all models</option>
          {modelOptions.map((modelId) => (
            <option key={modelId} value={modelId}>
              {modelId}
            </option>
          ))}
        </select>
      </label>
      <label>
        <span>game</span>
        <select value={selectedReplayIndex} onChange={(event) => onReplay(Number(event.currentTarget.value))}>
          {replayGames.length === 0 ? (
            <option value={0}>sample replay</option>
          ) : (
            replayGames.map((game, index) => (
              <option key={`${game.generation}-${game.modelId}-${game.opponentId}-${game.gameIndex}`} value={index}>
                {selectsActualGames
                  ? `G${game.generation} game ${game.gameIndex + 1}`
                  : `G${game.generation} ${game.modelId} vs ${game.opponentId} game ${game.gameIndex + 1}`}
              </option>
            ))
          )}
        </select>
      </label>
      <div className="replay-summary">
        reward={selectedReplay?.reward ?? 0}; full_replay={String(telemetry.fullReplay)};
        frame_status={fullGameFrameStatus}; stored_games={telemetry.storedReplayGames}; frames={replayFrameCount}
      </div>
    </section>
  );
}

function actualReplayGames(replayGames: ReplayGame[]) {
  const gamesByActualGame = new Map<string, ReplayGame>();
  for (const game of replayGames) {
    const key = actualReplayGameKey(game);
    if (!gamesByActualGame.has(key)) {
      gamesByActualGame.set(key, game);
    }
  }
  return Array.from(gamesByActualGame.values()).sort(
    (left, right) =>
      right.generation - left.generation ||
      left.gameIndex - right.gameIndex ||
      left.modelId.localeCompare(right.modelId),
  );
}

function Status({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className={`status-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function mergeReplayGames(previous: ReplayGame[], incoming: ReplayGame[]) {
  const byKey = new Map(previous.map((game) => [replayGameKey(game), game]));
  for (const game of incoming) {
    const key = replayGameKey(game);
    const existing = byKey.get(key);
    if (!existing || (existing.frames.length === 0 && game.frames.length > 0)) {
      byKey.set(key, game);
    }
  }
  return Array.from(byKey.values()).sort(
    (left, right) =>
      left.generation - right.generation ||
      left.modelId.localeCompare(right.modelId) ||
      left.gameIndex - right.gameIndex,
  );
}

function replayGameKey(game: ReplayGame) {
  return `${game.generation}:${game.modelId}:${game.opponentId}:${game.gameIndex}`;
}

function actualReplayGameKey(game: ReplayGame) {
  return `${game.generation}:${game.gameIndex}`;
}

function formatNumber(value: number) {
  return value.toLocaleString(STATUS_NUMBER_LOCALE, { maximumFractionDigits: 0 });
}

function formatSeconds(value: number) {
  return `${value.toFixed(3)} s`;
}
