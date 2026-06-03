import { useEffect, useRef } from 'react';
import type { ReplayFrame } from './types';

const BOARD_UNITS = 100;
const SUN_CENTER_UNITS = 50;
const SUN_RADIUS_UNITS = 10;
const COMET_TAIL_MAX_SEGMENTS = 5;
const SUN_GLOW_INNER_RATIO = 0.5;
const SUN_GLOW_OUTER_RATIO = 2.5;
const FLEET_SIZE_BASE = 0.4;
const FLEET_SIZE_LOG_SCALE = 2.0;
const FLEET_SIZE_REFERENCE_SHIPS = 1000;
const FLEET_WING_RATIO = 0.6;
const FLEET_NOTCH_RATIO = 0.3;
const PLAYER_LINE_ALPHA = 0.55;

const OWNER_COLORS = new Map<number, string>([
  [-1, '#666666'],
  [0, '#0072B2'],
  [1, '#D55E00'],
  [2, '#009E73'],
  [3, '#F0E442'],
]);

export function ReplayCanvas({
  frame,
  frames,
  frameIndex,
  frameCount,
  isPlaying,
  onFrameIndex,
  onTogglePlaying,
}: {
  frame: ReplayFrame;
  frames: ReplayFrame[];
  frameIndex: number;
  frameCount: number;
  isPlaying: boolean;
  onFrameIndex: (frameIndex: number) => void;
  onTogglePlaying: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext('2d');
    if (!context) {
      return;
    }
    const width = canvas.width;
    const height = canvas.height;
    const scale = Math.min(width, height) / BOARD_UNITS;
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#000000';
    context.fillRect(0, 0, width, height);

    const sunX = SUN_CENTER_UNITS * scale;
    const sunY = SUN_CENTER_UNITS * scale;
    const sunR = SUN_RADIUS_UNITS * scale;
    const glow = context.createRadialGradient(
      sunX,
      sunY,
      sunR * SUN_GLOW_INNER_RATIO,
      sunX,
      sunY,
      sunR * SUN_GLOW_OUTER_RATIO,
    );
    glow.addColorStop(0, 'rgba(255, 200, 50, 0.6)');
    glow.addColorStop(0.5, 'rgba(255, 150, 20, 0.2)');
    glow.addColorStop(1, 'rgba(255, 100, 0, 0)');
    context.fillStyle = glow;
    context.fillRect(0, 0, width, height);
    context.beginPath();
    context.arc(sunX, sunY, sunR, 0, Math.PI * 2);
    context.fillStyle = '#FFB800';
    context.fill();
    context.strokeStyle = '#FFD700';
    context.lineWidth = 1;
    context.stroke();

    drawCometTails(context, frame, scale);
    const cometPlanetIds = new Set(frame.comets.flatMap((group) => group.planetIds));

    for (const planet of frame.planets) {
      const color = OWNER_COLORS.get(planet.owner) ?? '#d7dde5';
      const px = planet.x * scale;
      const py = planet.y * scale;
      const pr = planet.radius * scale;
      context.beginPath();
      context.arc(px, py, pr, 0, Math.PI * 2);
      context.fillStyle = color;
      context.globalAlpha = planet.owner >= 0 ? 0.85 : 0.5;
      context.fill();
      context.globalAlpha = 1;
      context.beginPath();
      context.arc(px, py, pr, 0, Math.PI * 2);
      context.strokeStyle = cometPlanetIds.has(planet.id) ? '#88ccff' : '#555555';
      context.lineWidth = cometPlanetIds.has(planet.id) ? 2 : 1;
      context.stroke();
      drawProductionDots(context, planet.x, planet.y, planet.radius, planet.production, planet.owner, scale);
    }

    for (const fleet of frame.fleets) {
      drawFleet(context, fleet.owner, fleet.x, fleet.y, fleet.angle, fleet.ships, scale);
    }

    drawPlanetLabels(context, frame, scale);
    drawFleetLabels(context, frame, scale);

    context.font = `${Math.max(8, scale * 1.5)}px Inter, sans-serif`;
    context.textAlign = 'left';
    context.textBaseline = 'top';
    context.fillStyle = '#888888';
    context.fillText(`Step ${frame.step}`, 6, 6);
  }, [frame]);

  return (
    <div className="replay-panel">
      <div className="panel-header">
        <span>Replay canvas</span>
        <strong>
          step {frame.step} | frame {frameIndex + 1}/{frameCount}
        </strong>
      </div>
      <canvas ref={canvasRef} width={720} height={720} aria-label="Orbit Wars replay canvas" />
      <div className="replay-timeline-row">
        <button
          className={isPlaying ? 'playback-button active' : 'playback-button'}
          type="button"
          disabled={frameCount <= 1}
          onClick={onTogglePlaying}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <input
          className="timeline"
          type="range"
          min="0"
          max={Math.max(0, frameCount - 1)}
          value={frameIndex}
          aria-label="replay frame"
          onChange={(event) => onFrameIndex(Number(event.currentTarget.value))}
        />
      </div>
    </div>
  );
}

function drawCometTails(
  context: CanvasRenderingContext2D,
  frame: ReplayFrame,
  scale: number,
) {
  for (const group of frame.comets) {
    for (const path of group.paths) {
      const tailLength = Math.min(group.pathIndex + 1, path.length, COMET_TAIL_MAX_SEGMENTS);
      if (tailLength < 2) {
        continue;
      }
      for (let tailIndex = 1; tailIndex < tailLength; tailIndex += 1) {
        const pathIndex = group.pathIndex - tailIndex;
        if (pathIndex < 0) {
          break;
        }
        const alpha = 0.4 * (1 - tailIndex / tailLength);
        context.beginPath();
        context.moveTo(path[pathIndex + 1][0] * scale, path[pathIndex + 1][1] * scale);
        context.lineTo(path[pathIndex][0] * scale, path[pathIndex][1] * scale);
        context.strokeStyle = `rgba(200, 220, 255, ${alpha})`;
        context.lineWidth = ((2.5 - (1.5 * tailIndex) / tailLength) * scale) / 5;
        context.lineCap = 'round';
        context.stroke();
      }
    }
  }
}

function drawProductionDots(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  production: number,
  owner: number,
  scale: number,
) {
  if (owner < 0 || production <= 0) {
    return;
  }
  const dotRadius = Math.max(1, scale * 0.3);
  for (let dotIndex = 0; dotIndex < production; dotIndex += 1) {
    const dotAngle = (dotIndex / production) * Math.PI * 2 - Math.PI / 2;
    const dotDistance = radius * scale + dotRadius + 2;
    const dotX = x * scale + Math.cos(dotAngle) * dotDistance;
    const dotY = y * scale + Math.sin(dotAngle) * dotDistance;
    context.beginPath();
    context.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
    context.fillStyle = '#aaaaaa';
    context.fill();
  }
}

function drawFleet(
  context: CanvasRenderingContext2D,
  owner: number,
  x: number,
  y: number,
  angle: number,
  ships: number,
  scale: number,
) {
  const color = OWNER_COLORS.get(owner) ?? '#d7dde5';
  const size = (FLEET_SIZE_BASE + (FLEET_SIZE_LOG_SCALE * Math.log(Math.max(1, ships))) / Math.log(FLEET_SIZE_REFERENCE_SHIPS)) * scale;
  context.save();
  context.translate(x * scale, y * scale);
  context.rotate(angle);
  context.beginPath();
  context.moveTo(size, 0);
  context.lineTo(-size, -size * FLEET_WING_RATIO);
  context.lineTo(-size * FLEET_NOTCH_RATIO, 0);
  context.lineTo(-size, size * FLEET_WING_RATIO);
  context.closePath();
  context.fillStyle = color;
  context.globalAlpha = 0.85;
  context.fill();
  context.globalAlpha = 1;
  context.strokeStyle = '#222222';
  context.lineWidth = 0.5;
  context.stroke();
  context.strokeStyle = `rgba(255, 255, 255, ${PLAYER_LINE_ALPHA})`;
  context.lineWidth = size * 0.15;
  context.lineCap = 'round';
  if (owner === 1 || owner === 3) {
    context.beginPath();
    context.moveTo(size * 0.8, 0);
    context.lineTo(-size * 0.2, 0);
    context.stroke();
  }
  if (owner === 2 || owner === 3) {
    context.beginPath();
    context.moveTo(size * 0.6, -size * 0.15);
    context.lineTo(-size * 0.7, -size * 0.45);
    context.stroke();
    context.beginPath();
    context.moveTo(size * 0.6, size * 0.15);
    context.lineTo(-size * 0.7, size * 0.45);
    context.stroke();
  }
  context.restore();
}

function drawPlanetLabels(context: CanvasRenderingContext2D, frame: ReplayFrame, scale: number) {
  const planetFontSize = Math.max(8, scale * 1.8);
  context.font = `bold ${planetFontSize}px Inter, sans-serif`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  for (const planet of frame.planets) {
    const px = planet.x * scale;
    const py = planet.y * scale;
    const shipText = Math.floor(planet.ships).toString();
    context.fillStyle = '#000000';
    context.fillText(shipText, px + 0.5, py + 0.5);
    context.fillStyle = '#ffffff';
    context.fillText(shipText, px, py);
  }
}

function drawFleetLabels(context: CanvasRenderingContext2D, frame: ReplayFrame, scale: number) {
  const fleetFontSize = Math.max(6, scale * 1.2);
  context.font = `${fleetFontSize}px Inter, sans-serif`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  for (const fleet of frame.fleets) {
    const color = OWNER_COLORS.get(fleet.owner) ?? '#d7dde5';
    const labelOffset = fleet.y >= SUN_CENTER_UNITS ? -scale * 2.5 : scale * 2.5;
    context.fillStyle = color;
    context.fillText(Math.floor(fleet.ships).toString(), fleet.x * scale, fleet.y * scale + labelOffset);
  }
}
