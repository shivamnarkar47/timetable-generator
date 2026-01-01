from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import random
import copy
from collections import defaultdict
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TimetableSolver")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. INPUT MODELS
# ==========================================


class ConfigData(BaseModel):
    slots_per_day: int
    recess_index: int
    days: List[str]


class ResourceData(BaseModel):
    lab_rooms: List[str]
    theory_rooms: List[str]


class SubjectData(BaseModel):
    name: str
    code: str
    type: str
    weekly_load: int
    duration: Optional[int] = 1


class FacultyData(BaseModel):
    id: str
    name: str
    role: str
    experience: int
    shift: str
    skills: List[str] = []


class AllocationData(BaseModel):
    teacher_id: str
    subject_name: str
    division: str


class RoomInput(BaseModel):
    name: str
    type: str
    special_assignment: Optional[str] = None


class TimetableRequest(BaseModel):
    config: ConfigData
    resources: ResourceData
    subjects: Dict[str, List[SubjectData]]
    lab_prefs: Dict[str, List[str]]
    home_rooms: Dict[str, str]
    shift_bias: Dict[str, str] = {}
    faculty: List[FacultyData]
    allocations: List[AllocationData]
    divisions: Dict[str, List[str]]
    rooms: List[RoomInput]


# ==========================================
# 2. CORE CLASSES
# ==========================================


class Teacher:
    def __init__(self, data: FacultyData):
        self.id = data.id
        self.name = data.name
        self.shift = data.shift
        self.current_load = 0
        self.max_load = 20

    def assign_load(self, duration=1):
        self.current_load += duration

    def is_available(self, slot, total_slots):
        if self.shift == "A" and slot >= total_slots - 2:
            return False
        if self.shift == "B" and slot == 0:
            return False
        return True

    def __repr__(self):
        return self.name


class DummyTeacher:
    def __init__(self, id="-1", name="TBA"):
        self.id = id
        self.name = name
        self.current_load = 0
        self.max_load = 999
        self.shift = "ALL"

    def is_available(self, slot, total):
        return True

    def assign_load(self, duration=1):
        pass


class Gene:
    def __init__(
        self,
        div,
        type,
        subject,
        duration=1,
        teachers_list=None,
        lab_subjects=None,
        batch_ids=None,
    ):
        self.div = div
        self.type = type
        self.subject = subject
        self.duration = duration
        self.teachers_list = teachers_list if teachers_list else []
        self.lab_subjects = lab_subjects if lab_subjects else []
        self.batch_ids = batch_ids if batch_ids else []
        self.day = -1
        self.slot = -1
        self.assigned_rooms = []

    def __repr__(self):
        return f"{self.div}|{self.type}|{self.subject}"


class Schedule:
    def __init__(self, genes, constants):
        self.genes = genes
        self.constants = constants
        self.grid = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.div_slots = defaultdict(lambda: defaultdict(list))
        self.teacher_slots = defaultdict(lambda: defaultdict(list))
        self.div_batch_busy = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.theory_rooms_used = defaultdict(lambda: defaultdict(int))

        self.div_subjects = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        self.div_type_history = defaultdict(
            lambda: defaultdict(lambda: defaultdict(str))
        )
        self.div_daily_count = defaultdict(lambda: defaultdict(int))

    def is_free(self, day, start, gene, strict_repetition_check=True):
        if start + gene.duration > self.constants["SLOTS_PER_DAY"]:
            return False

        if strict_repetition_check:
            prev_s = start - 1
            if prev_s == self.constants["RECESS_INDEX"]:
                prev_s -= 1
            if prev_s >= 0:
                prev_sub = self.div_subjects[day][prev_s][gene.div]
                if prev_sub == gene.subject:
                    return False

            next_s = start + gene.duration
            if next_s == self.constants["RECESS_INDEX"]:
                next_s += 1
            if next_s < self.constants["SLOTS_PER_DAY"]:
                next_sub = self.div_subjects[day][next_s][gene.div]
                if next_sub == gene.subject:
                    return False

        for s in range(start, start + gene.duration):
            if s == self.constants["RECESS_INDEX"]:
                return False

            for b in gene.batch_ids:
                busy_batches = self.div_batch_busy[day][s][gene.div]
                if "ALL" in busy_batches:
                    return False
                if b == "ALL":
                    if len(busy_batches) > 0:
                        return False
                elif b in busy_batches:
                    return False

            for t in gene.teachers_list:
                if t.id != "-1":
                    if t.id in self.grid[day][s]["teacher"]:
                        return False
                    if not t.is_available(s, self.constants["SLOTS_PER_DAY"]):
                        return False

        return True

    def book(self, gene, day, start, rooms):
        gene.day = day
        gene.slot = start
        gene.assigned_rooms = rooms

        self.div_daily_count[gene.div][day] += 1

        for i in range(gene.duration):
            idx = start + i
            self.div_subjects[day][idx][gene.div] = gene.subject
            self.div_type_history[day][idx][gene.div] = gene.type

            for b in gene.batch_ids:
                self.div_batch_busy[day][idx][gene.div].add(b)
            for t in gene.teachers_list:
                if t.id != "-1":
                    self.grid[day][idx]["teacher"].add(t.id)
                    self.teacher_slots[t.id][day].append(idx)
            for r in rooms:
                if r != "TBA":
                    self.grid[day][idx]["room"].add(r)
            self.div_slots[gene.div][day].append(idx)
            if gene.type in ["THEORY", "ELECTIVE"]:
                self.theory_rooms_used[day][idx] += len(rooms)

    def calculate_gaps_and_sparse(self):
        gaps = 0
        sparse_penalty = 0
        for div, d_map in self.div_slots.items():
            for d, slots in d_map.items():
                slots.sort()
                valid = [s for s in slots if s != self.constants["RECESS_INDEX"]]
                if len(valid) > 1:
                    span = valid[-1] - valid[0] + 1
                    if valid[0] < self.constants["RECESS_INDEX"] < valid[-1]:
                        span -= 1
                    diff = span - len(valid)
                    if diff > 0:
                        gaps += diff

                daily_count = self.div_daily_count[div][d]
                if 0 < daily_count < 3:
                    sparse_penalty += 1
        return gaps, sparse_penalty


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================


def normalize_key(s):
    return re.sub(r"[^a-zA-Z0-9]", "", s).lower().replace("maths", "math")


def check_room_free(schedule, day, start, duration, room):
    for s in range(start, start + duration):
        if s == schedule.constants["RECESS_INDEX"]:
            return False
        if room in schedule.grid[day][s]["room"]:
            return False
    return True


def get_rooms_for_gene(
    schedule, day, start, gene, resources, home_rooms, special_rooms
):
    needed = len(gene.teachers_list)
    found_rooms = []

    if gene.type in ["THEORY", "ELECTIVE"]:
        if schedule.theory_rooms_used[day][start] + needed > len(
            resources.theory_rooms
        ):
            return None
        pool = list(resources.theory_rooms)
        random.shuffle(pool)
        home = home_rooms.get(gene.div)

        if home and home in pool:
            if check_room_free(schedule, day, start, gene.duration, home):
                pool.remove(home)
                pool.insert(0, home)

        for i in range(needed):
            assigned = None
            for r in pool:
                if r not in found_rooms and check_room_free(
                    schedule, day, start, gene.duration, r
                ):
                    assigned = r
                    break
            if assigned:
                found_rooms.append(assigned)

    else:
        reserved_rooms = {r for rooms in special_rooms.values() for r in rooms}
        lab_pool = [r for r in resources.lab_rooms if r not in reserved_rooms]
        random.shuffle(lab_pool)
        theory_pool = list(resources.theory_rooms)
        random.shuffle(theory_pool)

        for i in range(needed):
            sub_name = gene.lab_subjects[i]
            assigned = None

            if sub_name == "PROJECT" or sub_name == "LIBRARY":
                for r in theory_pool:
                    if r not in found_rooms and check_room_free(
                        schedule, day, start, gene.duration, r
                    ):
                        assigned = r
                        break
                if not assigned:
                    assigned = "Location TBA"
            else:
                norm_sub = normalize_key(sub_name)
                special_key = next(
                    (
                        k
                        for k in special_rooms
                        if normalize_key(k) in norm_sub or norm_sub in normalize_key(k)
                    ),
                    None,
                )

                if special_key:
                    candidates = special_rooms[special_key]
                    for r in candidates:
                        if r not in found_rooms and check_room_free(
                            schedule, day, start, gene.duration, r
                        ):
                            assigned = r
                            break
                    if not assigned:
                        return None
                else:
                    pool_to_use = theory_pool if gene.type == "MATHS_TUT" else lab_pool
                    for r in pool_to_use:
                        if r not in found_rooms and check_room_free(
                            schedule, day, start, gene.duration, r
                        ):
                            assigned = r
                            break

            if assigned:
                found_rooms.append(assigned)

    if len(found_rooms) == needed:
        return found_rooms
    return None


def calculate_cost(schedule, day, slot, gene, constants):
    cost = 0
    # 1. GRAVITY
    cost += slot * 100

    if "BE" in gene.div and slot >= 4:
        cost += 50000

    for t in gene.teachers_list:
        if t.id == "-1":
            continue
        t_slots = schedule.teacher_slots[t.id][day]
        prev, next_s = slot - 1, slot + gene.duration
        if prev == constants["RECESS_INDEX"]:
            prev -= 1
        if next_s == constants["RECESS_INDEX"]:
            next_s += 1
        consecutive = 0
        if prev in t_slots:
            consecutive += 1
        if next_s in t_slots:
            consecutive += 1
        if consecutive >= 1:
            cost += 1000
        if consecutive >= 2:
            cost += 5000

    # 4. NUCLEAR GAP CHECKER (Aggressive Update)
    current_slots = schedule.div_slots[gene.div][day]
    if current_slots:
        all_s = sorted(current_slots + [slot])

        # Calculate Span (including the potential new slot)
        span = all_s[-1] - all_s[0] + 1
        # Adjust for Recess
        if all_s[0] < constants["RECESS_INDEX"] < all_s[-1]:
            span -= 1

        count = len(all_s)
        actual_gaps = span - count

        if actual_gaps > 0:
            # 50 Million points per gap slot. Gap = Enemy #1.
            cost += actual_gaps * 50000000

            # The "Commuter Constraint": Spanning recess with a gap is instant death (200M)
            if (
                all_s[0] < constants["RECESS_INDEX"]
                and all_s[-1] > constants["RECESS_INDEX"]
            ):
                cost += 200000000
        else:
            # Reward compactness to break ties
            cost -= 10000

    if gene.type == "ELECTIVE":
        if slot == 0:
            cost -= 50000
        elif slot > 1:
            cost += 50000

    if gene.type == "MATHS_TUT":
        if slot >= constants["SLOTS_PER_DAY"] - 2:
            cost -= 100000
        elif slot < 5:
            cost += 50000

    if gene.type == "THEORY":
        prev1 = slot - 1
        if prev1 == constants["RECESS_INDEX"]:
            prev1 -= 1
        prev2 = prev1 - 1
        if prev2 == constants["RECESS_INDEX"]:
            prev2 -= 1
        if prev1 >= 0 and prev2 >= 0:
            t1 = schedule.div_type_history[day][prev1][gene.div]
            t2 = schedule.div_type_history[day][prev2][gene.div]
            if t1 == "THEORY" and t2 == "THEORY":
                cost += 5000

    if gene.type == "LAB":
        busy_batches = schedule.div_batch_busy[day][slot][gene.div]
        if busy_batches:
            cost -= 5000

    prev_s = slot - 1
    if prev_s == constants["RECESS_INDEX"]:
        prev_s -= 1
    if prev_s >= 0:
        prev_sub = schedule.div_subjects[day][prev_s][gene.div]
        if prev_sub == gene.subject:
            cost += 100000

    return cost


def solve(genes, config, resources, home_rooms, special_rooms):
    logger.info("--- Starting Solver (Zero Gap Aggression) ---")
    best_sched = None
    best_score = -float("inf")

    CONSTANTS = {"SLOTS_PER_DAY": config.slots_per_day, "RECESS_INDEX": 4}

    random.shuffle(genes)
    genes.sort(
        key=lambda g: 0
        if g.type == "LAB"
        else (1 if g.type == "MATHS_TUT" else (2 if g.type == "ELECTIVE" else 3))
    )

    TOTAL_BATCHES = 3

    for run in range(5000):
        schedule = Schedule(copy.deepcopy(genes), CONSTANTS)
        unplaced = []

        panic_mode = run > 1500
        strict_rep = run < 2500

        for g in genes:
            best_move = None
            min_cost = float("inf")

            days = list(range(len(config.days)))
            random.shuffle(days)
            all_slots = list(range(config.slots_per_day))

            valid_starts = []

            if g.duration == 2:
                hod_blocks = [0, 2, 5, 7]
                valid_hod = [s for s in hod_blocks if s + 2 <= config.slots_per_day]

                if len(g.batch_ids) < TOTAL_BATCHES:
                    valid_starts = sorted(valid_hod, key=lambda x: -x)
                else:
                    random.shuffle(valid_hod)
                    valid_starts = valid_hod

            elif g.type == "MATHS_TUT":
                late = [7, 8, 6]
                random.shuffle(late)
                others = [s for s in all_slots if s not in late and s != 3 and s != 4]
                random.shuffle(others)
                valid_starts = late + others
            elif g.type == "ELECTIVE":
                early = [0, 1]
                others = [s for s in all_slots if s not in early and s != 3 and s != 4]
                random.shuffle(others)
                valid_starts = early + others
            else:
                gap_filler = [3]
                others = [s for s in all_slots if s != 3 and s != 4]
                random.shuffle(others)
                valid_starts = gap_filler + others

            for d in days:
                for s in valid_starts:
                    if schedule.is_free(d, s, g, strict_repetition_check=strict_rep):
                        rooms = get_rooms_for_gene(
                            schedule, d, s, g, resources, home_rooms, special_rooms
                        )
                        if rooms:
                            cost = calculate_cost(schedule, d, s, g, CONSTANTS)
                            if cost < min_cost:
                                min_cost = cost
                                best_move = (d, s, rooms)
                                if panic_mode:
                                    break
                                if cost <= -100000:
                                    break
                if best_move and (panic_mode or min_cost <= -100000):
                    break

            if best_move:
                schedule.book(g, best_move[0], best_move[1], best_move[2])
            else:
                unplaced.append(g)

        score = 1000000
        score -= len(unplaced) * 100000000

        gaps, sparse_days = schedule.calculate_gaps_and_sparse()
        score -= gaps * 50000000  # Increased to match cost logic
        score -= sparse_days * 300000

        if run % 500 == 0:
            logger.info(
                f"Run {run}: Score={score} Unplaced={len(unplaced)} Gaps={gaps} Sparse={sparse_days}"
            )

        if score > best_score:
            best_score = score
            best_sched = schedule
            if len(unplaced) == 0 and gaps <= 3 and sparse_days == 0:
                break

    return best_sched


# ==========================================
# 4. API ENDPOINT
# ==========================================


@app.post("/generate-timetable")
async def generate_timetable(req: TimetableRequest):
    """
    Generate an optimized academic timetable based on the provided configuration.

    This endpoint takes comprehensive scheduling data including faculty, subjects,
    room allocations, and constraints, then uses a constraint-based optimization
    algorithm to generate a conflict-free timetable that minimizes gaps and
    maximizes resource utilization.

    Args:
        req (TimetableRequest): Complete timetable configuration data including:
            - config: Timing configuration (slots, recess, days)
            - resources: Available theory and lab rooms
            - subjects: Subject definitions by year
            - faculty: Teacher information and preferences
            - allocations: Teacher-subject-division assignments
            - rooms: Room details with special assignments
            - divisions: Student division structure
            - lab_prefs: Laboratory preferences
            - home_rooms: Preferred rooms for divisions
            - shift_bias: Teacher shift preferences

    Returns:
        dict: Generated timetable organized by division and day, containing
              session details with teacher, room, and timing information.

    Example Request:
        {
          "config": {
            "slots_per_day": 9,
            "recess_index": 4,
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
          },
          "resources": {
            "lab_rooms": ["Lab1", "Lab2", "Lab3"],
            "theory_rooms": ["Room101", "Room102", "Room103"]
          },
          "subjects": {
            "SE": [
              {
                "name": "Data Structures",
                "code": "CS201",
                "type": "Theory",
                "weekly_load": 3,
                "duration": 1
              }
            ]
          },
          "faculty": [
            {
              "id": "T001",
              "name": "Dr. Smith",
              "role": "Professor",
              "experience": 10,
              "shift": "A",
              "skills": ["Data Structures", "Algorithms"]
            }
          ],
          "allocations": [
            {
              "teacher_id": "T001",
              "subject_name": "Data Structures",
              "division": "SE-A"
            }
          ],
          "divisions": {
            "SE": ["SE-A", "SE-B"]
          },
          "rooms": [
            {
              "name": "Room101",
              "type": "theory",
              "special_assignment": null
            }
          ],
          "lab_prefs": {},
          "home_rooms": {"SE-A": "Room101"}
        }
    """
    teachers_map = {t.id: Teacher(t) for t in req.faculty}
    special_rooms = defaultdict(list)
    for r in req.rooms:
        if r.special_assignment:
            special_rooms[r.special_assignment].append(r.name)

    genes = []
    all_subjects_flat = []
    for year_list in req.subjects.values():
        all_subjects_flat.extend(year_list)

    div_allocs = defaultdict(lambda: defaultdict(list))

    for alloc in req.allocations:
        if not alloc.teacher_id:
            continue
        parts = alloc.division.split("-")

        if len(parts) >= 3:
            base_div = f"{parts[0]}-{parts[1]}"
            batch_id = parts[2].replace(parts[1], "")
            div_allocs[base_div]["LABS"].append(
                {
                    "batch": batch_id,
                    "subject": alloc.subject_name,
                    "teacher_id": alloc.teacher_id,
                }
            )
        elif re.search(r"\d$", alloc.division):
            base_div = alloc.division[:-1]
            batch_id = alloc.division[-1]
            div_allocs[base_div]["LABS"].append(
                {
                    "batch": batch_id,
                    "subject": alloc.subject_name,
                    "teacher_id": alloc.teacher_id,
                }
            )
        else:
            div_allocs[alloc.division]["THEORY"].append(
                {"subject": alloc.subject_name, "teacher_id": alloc.teacher_id}
            )

    for div, types in div_allocs.items():
        # A. LABS & TUTORIALS
        lab_entries = types["LABS"]
        batch_buckets = defaultdict(list)
        for entry in lab_entries:
            batch_buckets[entry["batch"]].append(entry)

        dur1_groups = defaultdict(list)
        dur2_groups = defaultdict(list)

        for b_id, entries in batch_buckets.items():
            for entry in entries:
                s_info = next(
                    (s for s in all_subjects_flat if s.name == entry["subject"]), None
                )
                if not s_info and not entry["subject"].lower().endswith("tut"):
                    continue

                dur = 2
                if s_info:
                    if s_info.type == "Tutorial" or s_info.weekly_load == 1:
                        dur = 1
                    elif s_info.type == "Lab":
                        dur = s_info.duration if s_info.duration else 2

                if entry["subject"].lower().endswith("tut"):
                    dur = 1

                if dur == 1:
                    dur1_groups[b_id].append(entry)
                else:
                    dur2_groups[b_id].append(entry)

        # 1. LABS (2H) - SMART CHUNKING
        batch_keys = sorted(dur2_groups.keys())
        if batch_keys:
            unique_lab_subjects = set()
            for b_k in batch_keys:
                for entry in dur2_groups[b_k]:
                    unique_lab_subjects.add(entry["subject"])

            capacity = len(unique_lab_subjects)
            if capacity == 0:
                capacity = 1

            max_len = max(len(dur2_groups[k]) for k in batch_keys)

            for i in range(max_len):
                current_step_allocations = []
                for k_idx, b_key in enumerate(batch_keys):
                    items = dur2_groups[b_key]
                    if not items:
                        continue
                    item_idx = (i + k_idx) % len(items)
                    entry = items[item_idx]
                    current_step_allocations.append(
                        {
                            "batch": b_key,
                            "subject": entry["subject"],
                            "teacher": teachers_map.get(
                                entry["teacher_id"], DummyTeacher()
                            ),
                        }
                    )

                for chunk_start in range(0, len(current_step_allocations), capacity):
                    chunk = current_step_allocations[
                        chunk_start : chunk_start + capacity
                    ]

                    if chunk:
                        g_subs = [x["subject"] for x in chunk]
                        g_batches = [x["batch"] for x in chunk]
                        g_teachers = [x["teacher"] for x in chunk]

                        genes.append(
                            Gene(
                                div,
                                "LAB",
                                "Session",
                                duration=2,
                                teachers_list=g_teachers,
                                lab_subjects=g_subs,
                                batch_ids=g_batches,
                            )
                        )

        # 2. TUTORIALS (1H)
        for b_id, items in dur1_groups.items():
            for entry in items:
                t = teachers_map.get(entry["teacher_id"], DummyTeacher())
                genes.append(
                    Gene(
                        div,
                        "MATHS_TUT",
                        entry["subject"],
                        duration=1,
                        teachers_list=[t],
                        lab_subjects=[entry["subject"]],
                        batch_ids=[entry["batch"]],
                    )
                )

        # B. THEORY
        electives = defaultdict(list)
        theory_list = []
        for item in types["THEORY"]:
            s_info = next(
                (s for s in all_subjects_flat if s.name == item["subject"]), None
            )
            if not s_info:
                continue
            t = teachers_map.get(item["teacher_id"], DummyTeacher())
            t.assign_load(s_info.weekly_load)
            if s_info.type == "Elective":
                electives[item["subject"]].append(t)
            else:
                theory_list.append((item["subject"], t, s_info))

        for sub_name, teacher, s_info in theory_list:
            for _ in range(s_info.weekly_load):
                genes.append(
                    Gene(
                        div,
                        "THEORY",
                        sub_name,
                        duration=1,
                        teachers_list=[teacher],
                        batch_ids=["ALL"],
                    )
                )

        if electives:
            max_load = 0
            elec_subjects = list(electives.keys())
            elec_teachers = []
            for sub in elec_subjects:
                s_info = next((s for s in all_subjects_flat if s.name == sub), None)
                load = s_info.weekly_load if s_info else 3
                max_load = max(max_load, load)
                elec_teachers.append(electives[sub][0])
            for _ in range(max_load):
                g = Gene(
                    div,
                    "ELECTIVE",
                    "Elective Block",
                    duration=1,
                    teachers_list=elec_teachers,
                    lab_subjects=elec_subjects,
                    batch_ids=["ALL"],
                )
                genes.append(g)

    # --- RUN SOLVER ---
    schedule = solve(genes, req.config, req.resources, req.home_rooms, special_rooms)

    if not schedule:
        raise HTTPException(status_code=500, detail="Unable to generate schedule")

    output = defaultdict(lambda: defaultdict(list))
    days_lookup = req.config.days

    for g in schedule.genes:
        if g.day == -1:
            continue
        entry = {
            "slot": g.slot,
            "duration": g.duration,
            "type": g.type,
            "subject": g.subject,
            "teacher": "TBA",
            "room": "TBA",
        }

        if g.type in ["LAB", "MATHS_TUT"]:
            entry["batches"] = []
            for i, sub in enumerate(g.lab_subjects):
                t_name = g.teachers_list[i].name if i < len(g.teachers_list) else "TBA"
                r_name = g.assigned_rooms[i] if i < len(g.assigned_rooms) else "TBA"
                b_id = g.batch_ids[i] if i < len(g.batch_ids) else "?"
                entry["batches"].append(
                    {
                        "batch": f"B{b_id}",
                        "subject": sub,
                        "teacher": t_name,
                        "room": r_name,
                    }
                )
            entry["subject"] = " / ".join(set(g.lab_subjects))
            entry["teacher"] = "Multiple"
            entry["room"] = "Multiple"

        elif g.type == "ELECTIVE":
            entry["subject"] = " / ".join(g.lab_subjects)
            entry["teacher"] = " / ".join([t.name for t in g.teachers_list])
            entry["room"] = " / ".join(g.assigned_rooms)

        else:  # THEORY
            entry["teacher"] = g.teachers_list[0].name
            entry["room"] = g.assigned_rooms[0]

        output[g.div][days_lookup[g.day]].append(entry)

    return output
