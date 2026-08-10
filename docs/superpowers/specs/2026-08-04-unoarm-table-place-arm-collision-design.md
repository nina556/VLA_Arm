# Table-Place IK: Arm–Table Collision Box + Grasp Orientation

**Date:** 2026-08-04  
**Status:** Approved (user: execute directly)  
**Extends:** `2026-08-04-unoarm-table-place-ik-datagen-design.md`

## Goal

Generated table_place Reach-IK episodes must keep the **entire right arm** out of a **table collision box**, use **lift → horizontal transport → lower** after grasp, and close the gripper only in **horizontal** or **vertical** TCP orientation (random per attempt).

## Decisions

| Topic | Choice |
|-------|--------|
| Collision body | Table AABB from `TABLE_PLACE_POS` / `TABLE_PLACE_HALF`, optional margin; check via geom distance to `table_geom` (no penetration) for all `Right_*` body geoms |
| Peg | Resting contact OK before lift; after attach, peg follows arm rules via peg-bottom clearance on lift/transport only |
| Path after grasp | Lift to `lift_z` → XY-only hops at ~constant high Z → lower to `place_z` → release |
| Grasp orientation | Random `horizontal` (+X ≈ −Y) or `vertical` (+X ≈ −Z); locked from approach through close |
| Enforcement | Reject IK keyframes that collide; discard episodes if any interpolated step arm-collides |
| Sword Reach-IK | Unchanged |

## Non-goals

- Collision-repulsive DLS inside `solve_right_tcp_ik`
- Randomizing peg/circle pose
