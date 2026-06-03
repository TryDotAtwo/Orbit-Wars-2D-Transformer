export type Planet = {
  id: number;
  owner: number;
  x: number;
  y: number;
  radius: number;
  ships: number;
  production: number;
};

export type Fleet = {
  id: number;
  owner: number;
  x: number;
  y: number;
  angle: number;
  ships: number;
};

export type ReplayFrame = {
  step: number;
  planets: Planet[];
  fleets: Fleet[];
  comets: ReplayCometGroup[];
};

export type ReplayCometGroup = {
  planetIds: number[];
  paths: Array<Array<[number, number]>>;
  pathIndex: number;
};

export type MetricPoint = {
  generation: number;
  winRate: number;
  gamesPerSecond: number;
  turnsPerSecond: number;
  p95LatencyMs: number;
  gpuUtilization: number;
  evaluatedGames: number;
  sampledReplayGames: number;
  modelActionCalls: number;
  launchActions: number;
  launchedShips: number;
  captures: number;
  fleetHits: number;
  hitShips: number;
  sunDestroyedFleets: number;
  sunDestroyedShips: number;
  avgFleetSize: number;
  avgLaunchActionsPerTurn: number;
  avgLaunchedShipsPerTurn: number;
  avgModelActionMs: number;
  inferenceBatchCalls: number;
  maxInferenceBatchSize: number;
  simultaneousGames: number;
  modelActionSeconds: number;
  simulationStepSeconds: number;
  evaluationSeconds: number;
  replaySeconds: number;
  replayWriteSeconds: number;
  generationValidationGames: number;
  generationValidationSeconds: number;
  backpropSamples: number;
  backpropModels: number;
  backpropSeconds: number;
  reproductionSeconds: number;
  generationSeconds: number;
};

export type GenerationWinRatePoint = {
  validationGeneration: number;
  evaluatedGeneration: number;
  modelCount: number;
  games: number;
  wins: number;
  draws: number;
  losses: number;
  winRate: number;
};

export type ModelRow = {
  id: string;
  parent: string;
  rating: number;
  wins: number;
  draws: number;
  losses: number;
  games: number;
  captures: number;
  fleetHits: number;
  hitShips: number;
  sunDestroyedFleets: number;
  sunDestroyedShips: number;
  outOfBoundsFleets: number;
  outOfBoundsShips: number;
  launchActions: number;
  launchedShips: number;
  avgFleetSize: number;
  mutation: string;
  selected: boolean;
};

export type ReplayGame = {
  generation: number;
  modelId: string;
  gameIndex: number;
  opponentId: string;
  reward: number;
  frames: ReplayFrame[];
  gamePath?: string;
  indexed?: boolean;
};

export type ReplayGameIndexEntry = ReplayGame & {
  gamePath: string;
  indexed: true;
};

export type ReplayChunk = {
  generation: number;
  path: string;
  gameCount: number;
  actualGameCount?: number;
};

export type DashboardTelemetry = {
  source: 'sample' | 'artifact';
  sourceMessage: string;
  runId: string;
  activeGeneration: number;
  runProfile: string;
  trainingMode: string;
  strictGameRules: boolean;
  episodeSteps: number;
  expectedFramesPerGame: number;
  populationSize: number;
  eliteCount: number;
  gamesPerModel: number;
  replaysPerModel: number;
  generationReplayGameCount: number;
  playersPerGame: number;
  simultaneousGames: number;
  inferenceBatchCalls: number;
  maxInferenceBatchSize: number;
  mapSource: string;
  turnLoop: string;
  fullReplay: boolean;
  replayFrameStride: number;
  liveReplayFormat?: string;
  liveReplayPath?: string;
  liveReplayFrameCount?: number;
  storedReplayGames: number;
  gamesPerSecond: number;
  turnsPerSecond: number;
  gpuUtilization: number;
  cpuWorkers: number;
  evalQueue: number;
  submissionGate: string;
  metrics: MetricPoint[];
  generationWinRates: GenerationWinRatePoint[];
  models: ModelRow[];
  frames: ReplayFrame[];
  replayChunks: ReplayChunk[];
  replayGames: ReplayGame[];
  validationErrors: string[];
  warnings: string[];
};
