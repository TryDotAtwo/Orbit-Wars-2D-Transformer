#!/usr/bin/env python3
import argparse
import os
import struct


SLOT_MAGIC = b"OWSLOT1\n"
LIVE_MAGIC = b"OWLIVE1\n"
RECORD_HEADER = 1
RECORD_FRAME = 2
RECORD_RESULT = 3
U32_BYTES = 4
I32_BYTES = 4


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
        return struct.unpack("<I", self.read(U32_BYTES))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> bytes:
        size = self.u32()
        return self.read(size)


def push_u32(out: bytearray, value: int) -> None:
    out.extend(struct.pack("<I", value))


def parse_header(data: bytes):
    reader = Reader(data)
    if reader.read(len(SLOT_MAGIC)) != SLOT_MAGIC:
        raise ValueError("bad_slot_magic")
    generation = reader.u32()
    active_player_count = reader.u32()
    game_count = reader.u32()
    frame_slots_per_game = reader.u32()
    frame_slot_bytes = reader.u32()
    frame_base_offset = reader.u64()
    games = []
    for _ in range(game_count):
        game_index = reader.u32()
        participant_count = reader.u32()
        participants = []
        for _ in range(participant_count):
            model_index = reader.u32()
            opponent_label = reader.string()
            participants.append((model_index, opponent_label))
        games.append((game_index, participants))
    return generation, active_player_count, frame_slots_per_game, frame_slot_bytes, frame_base_offset, games


def compact_slot(source: str, target: str, remove_source: bool) -> None:
    with open(source, "rb") as handle:
        data = handle.read()
    generation, active_player_count, frame_slots_per_game, frame_slot_bytes, frame_base_offset, games = parse_header(data)
    out = bytearray(LIVE_MAGIC)
    out.append(RECORD_HEADER)
    push_u32(out, generation)
    push_u32(out, active_player_count)
    push_u32(out, len(games))
    for game_index, participants in games:
        push_u32(out, game_index)
        push_u32(out, len(participants))
        for model_index, opponent_label in participants:
            push_u32(out, model_index)
            push_u32(out, len(opponent_label))
            out.extend(opponent_label)

    per_slot_bytes = U32_BYTES + frame_slot_bytes
    result_base_offset = frame_base_offset + len(games) * frame_slots_per_game * per_slot_bytes
    for game_offset, (game_index, _) in enumerate(games):
        for frame_slot in range(frame_slots_per_game):
            slot_offset = frame_base_offset + (game_offset * frame_slots_per_game + frame_slot) * per_slot_bytes
            frame_length = struct.unpack("<I", data[slot_offset : slot_offset + U32_BYTES])[0]
            if frame_length == 0:
                continue
            if frame_length > frame_slot_bytes:
                raise ValueError(f"frame_too_large game={game_index} slot={frame_slot} length={frame_length}")
            out.append(RECORD_FRAME)
            push_u32(out, game_index)
            payload_start = slot_offset + U32_BYTES
            out.extend(data[payload_start : payload_start + frame_length])

    for game_offset, (game_index, participants) in enumerate(games):
        result_offset = result_base_offset + game_offset * (U32_BYTES + len(participants) * I32_BYTES)
        reward_count = struct.unpack("<I", data[result_offset : result_offset + U32_BYTES])[0]
        if reward_count > len(participants):
            raise ValueError(f"reward_count_too_large game={game_index} count={reward_count}")
        out.append(RECORD_RESULT)
        push_u32(out, game_index)
        push_u32(out, reward_count)
        rewards_start = result_offset + U32_BYTES
        out.extend(data[rewards_start : rewards_start + reward_count * I32_BYTES])

    tmp_target = f"{target}.tmp"
    with open(tmp_target, "wb") as handle:
        handle.write(out)
    os.replace(tmp_target, target)
    if remove_source:
        os.remove(source)
    print(f"compacted source={source} target={target} games={len(games)} bytes={len(out)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--remove-source", action="store_true")
    args = parser.parse_args()
    compact_slot(args.source, args.target, args.remove_source)


if __name__ == "__main__":
    main()
