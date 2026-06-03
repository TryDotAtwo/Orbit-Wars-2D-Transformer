#!/usr/bin/env python3
import argparse
import os
import struct
from pathlib import Path


LIVE_MAGIC = b"OWLIVE1\n"
RECORD_HEADER = 1
RECORD_FRAME = 2
RECORD_RESULT = 3


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read(self, size: int) -> bytes:
        chunk = self.data[self.offset : self.offset + size]
        if len(chunk) != size:
            raise ValueError("unexpected_eof")
        self.offset += size
        return chunk

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]


def push_u32(out: bytearray, value: int) -> None:
    out.extend(struct.pack("<I", value))


def parse_header(reader: Reader):
    if reader.read(len(LIVE_MAGIC)) != LIVE_MAGIC:
        raise ValueError("bad_live_magic")
    record_type = reader.read(1)[0]
    if record_type != RECORD_HEADER:
        raise ValueError("missing_header_record")
    generation = reader.u32()
    active_player_count = reader.u32()
    game_count = reader.u32()
    games = []
    for _ in range(game_count):
        game_index = reader.u32()
        participant_count = reader.u32()
        participants = []
        for _ in range(participant_count):
            model_index = reader.u32()
            label_size = reader.u32()
            label = reader.read(label_size)
            participants.append((model_index, label))
        games.append((game_index, participants))
    return generation, active_player_count, games


def copy_payload(reader: Reader, out: bytearray, keep_games: set[int], record_type: int) -> None:
    game_index = reader.u32()
    if record_type == RECORD_FRAME:
        frame_start = reader.offset
        reader.u32()
        planet_count = reader.u32()
        for _ in range(planet_count):
            reader.read(28)
        fleet_count = reader.u32()
        for _ in range(fleet_count):
            reader.read(24)
        comet_count = reader.u32()
        for _ in range(comet_count):
            comet_planet_count = reader.u32()
            reader.read(comet_planet_count * 4)
            comet_path_count = reader.u32()
            for _ in range(comet_path_count):
                point_count = reader.u32()
                reader.read(point_count * 8)
            reader.read(4)
        payload = reader.data[frame_start : reader.offset]
        if game_index in keep_games:
            out.append(RECORD_FRAME)
            push_u32(out, game_index)
            out.extend(payload)
        return
    if record_type == RECORD_RESULT:
        reward_count = reader.u32()
        rewards = reader.read(reward_count * 4)
        if game_index in keep_games:
            out.append(RECORD_RESULT)
            push_u32(out, game_index)
            push_u32(out, reward_count)
            out.extend(rewards)
        return
    raise ValueError(f"unknown_record_type={record_type}")


def trim_owlive(path: Path, max_games: int) -> tuple[int, int, int, int]:
    data = path.read_bytes()
    reader = Reader(data)
    generation, active_player_count, games = parse_header(reader)
    kept_games = games[:max_games]
    keep_indices = {game_index for game_index, _ in kept_games}
    if len(games) <= max_games:
        return generation, len(games), len(games), len(data)

    out = bytearray(LIVE_MAGIC)
    out.append(RECORD_HEADER)
    push_u32(out, generation)
    push_u32(out, active_player_count)
    push_u32(out, len(kept_games))
    for game_index, participants in kept_games:
        push_u32(out, game_index)
        push_u32(out, len(participants))
        for model_index, label in participants:
            push_u32(out, model_index)
            push_u32(out, len(label))
            out.extend(label)

    while reader.offset < len(reader.data):
        record_type = reader.read(1)[0]
        copy_payload(reader, out, keep_indices, record_type)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(out)
    os.replace(tmp_path, path)
    return generation, len(games), len(kept_games), len(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--max-games", type=int, required=True)
    args = parser.parse_args()
    if args.max_games < 1:
        raise ValueError("max_games_must_be_positive")
    for raw_path in args.paths:
        path = Path(raw_path)
        generation, old_count, new_count, new_bytes = trim_owlive(path, args.max_games)
        print(
            f"trimmed path={path} generation={generation} old_games={old_count} "
            f"new_games={new_count} bytes={new_bytes}"
        )


if __name__ == "__main__":
    main()
