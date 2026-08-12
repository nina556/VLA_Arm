import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import URDFLoader from "urdf-loader";

export class RobotScene {
  constructor(canvas, { onSystemMessage = () => {} } = {}) {
    this.canvas = canvas;
    this.onSystemMessage = onSystemMessage;
    this.loadState = document.getElementById("sceneLoadState");
    this.modelState = document.getElementById("modelState");
    this.fpsLabel = document.getElementById("fpsLabel");

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xe8eef2);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.18;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.minDistance = 0.8;
    this.controls.maxDistance = 10;
    this.controls.target.set(0, 1.0, 0);

    this.robot = null;
    this.jointTargets = new Map();
    this.jointDisplay = new Map();
    this.frameCounter = 0;
    this.frameWindowStart = performance.now();
    this.lastAnimationTime = this.frameWindowStart;
    this.timeOfDay = "day";
    this._todBusy = false;
    this._bgDay = new THREE.Color(0xe8eef2);
    this._bgNight = new THREE.Color(0x0a0c10);
    this._blackoutEl = document.getElementById("sceneBlackout");

    this.addLighting();
    this.applyDayLookImmediate();
    this.addGround();
    this.setDefaultView();
    this.loadRobot();
    this.resize();
    window.addEventListener("resize", () => this.resize());
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.animate();
  }

  addLighting() {
    // --- Day: original bright key / fill / rim ---
    this.dayLightGroup = new THREE.Group();
    this.dayLightGroup.name = "DayLights";
    const key = new THREE.DirectionalLight(0xffffff, 1.75);
    key.position.set(3.5, 5.5, 4);
    key.castShadow = true;
    key.shadow.mapSize.setScalar(2048);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 15;
    key.shadow.camera.left = -4;
    key.shadow.camera.right = 4;
    key.shadow.camera.top = 4;
    key.shadow.camera.bottom = -4;
    key.userData.baseIntensity = 1.75;
    this.dayLightGroup.add(key);

    const fill = new THREE.DirectionalLight(0xfff2d6, 0.7);
    fill.position.set(-4, 3, -2.5);
    fill.userData.baseIntensity = 0.7;
    this.dayLightGroup.add(fill);

    const rim = new THREE.DirectionalLight(0xd6e8ff, 0.45);
    rim.position.set(0, 2.5, -4);
    rim.userData.baseIntensity = 0.45;
    this.dayLightGroup.add(rim);

    const dayAmb = new THREE.AmbientLight(0xffffff, 0.62);
    dayAmb.userData.baseIntensity = 0.62;
    this.dayLightGroup.add(dayAmb);
    this.scene.add(this.dayLightGroup);

    // --- Night fill (very soft) ---
    this.nightAmb = new THREE.AmbientLight(0x1a2030, 0.14);
    this.nightAmb.userData.baseIntensity = 0.14;
    this.nightHemi = new THREE.HemisphereLight(0x2a3344, 0x080a0e, 0.22);
    this.nightHemi.userData.baseIntensity = 0.22;
    this.scene.add(this.nightAmb);
    this.scene.add(this.nightHemi);

    this.beamGroup = new THREE.Group();
    this.beamGroup.name = "VolumetricBeams";
    this.scene.add(this.beamGroup);

    const robotAim = new THREE.Vector3(0, 1.05, 0);
    const robotSpots = [
      { pos: [2.6, 4.6, 3.0], color: 0xfff0d8, intensity: 22 },
      { pos: [-2.4, 4.4, 2.7], color: 0xdde8ff, intensity: 18 },
      { pos: [0.15, 5.0, -2.2], color: 0xfff6e8, intensity: 16 },
    ];
    this.robotSpotLights = [];
    this.robotBeams = [];
    robotSpots.forEach((spec, i) => {
      const angle = Math.PI / 6.5;
      const spot = this.createSpotlight({
        color: spec.color,
        intensity: spec.intensity,
        position: spec.pos,
        target: robotAim,
        distance: 14,
        angle,
        penumbra: 0.82,
        castShadow: i === 0,
      });
      spot.name = `RobotSpot${i}`;
      spot.userData.baseIntensity = spec.intensity;
      this.robotSpotLights.push(spot);

      const beam = this.createCircularBeam({
        color: spec.color,
        from: new THREE.Vector3(...spec.pos),
        to: robotAim.clone(),
        angle,
        opacity: 0.09,
      });
      beam.name = `RobotBeam${i}`;
      this.beamGroup.add(beam);
      this.robotBeams.push(beam);
    });
  }

  applyDayLookImmediate() {
    this.timeOfDay = "day";
    this.setDayLightsLevel(1);
    this.setNightFillLevel(0);
    this.setRobotLightsLevel(0);
    this.dayLightGroup.visible = true;
    this.beamGroup.visible = false;
    this.scene.background.copy(this._bgDay);
    this.renderer.toneMappingExposure = 1.18;
    this.setBlackout(0);
    document.body.classList.remove("tod-night");
    this.syncTodButton();
  }

  applyNightLookImmediate() {
    this.timeOfDay = "night";
    this.setDayLightsLevel(0);
    this.dayLightGroup.visible = false;
    this.setNightFillLevel(1);
    this.setRobotLightsLevel(1);
    this.beamGroup.visible = true;
    this.scene.background.copy(this._bgNight);
    this.renderer.toneMappingExposure = 1.05;
    this.setBlackout(0);
    document.body.classList.add("tod-night");
    this.syncTodButton();
  }

  setDayLightsLevel(t) {
    this.dayLightGroup?.traverse((obj) => {
      if (obj.isLight && obj.userData.baseIntensity != null) {
        obj.intensity = obj.userData.baseIntensity * t;
      }
    });
  }

  setNightFillLevel(t) {
    if (this.nightAmb) this.nightAmb.intensity = this.nightAmb.userData.baseIntensity * t;
    if (this.nightHemi) this.nightHemi.intensity = this.nightHemi.userData.baseIntensity * t;
  }

  setRobotLightsLevel(t) {
    (this.robotSpotLights || []).forEach((spot) => {
      spot.intensity = (spot.userData.baseIntensity || 0) * t;
      spot.visible = t > 0.001;
    });
    (this.robotBeams || []).forEach((beam) => {
      beam.visible = t > 0.001;
      beam.traverse((obj) => {
        if (obj.isMesh && obj.material && obj.userData.baseOpacity != null) {
          obj.material.opacity = obj.userData.baseOpacity * t;
        }
      });
    });
  }

  setBlackout(t) {
    if (!this._blackoutEl) return;
    this._blackoutEl.style.opacity = String(Math.max(0, Math.min(1, t)));
  }

  syncTodButton() {
    const btn = document.getElementById("todToggleBtn");
    if (!btn) return;
    const night = this.timeOfDay === "night";
    btn.dataset.mode = this.timeOfDay;
    btn.title = night ? "切换到白昼" : "切换到黑夜";
    btn.setAttribute("aria-label", btn.title);
    btn.textContent = night ? "昼" : "夜";
    btn.disabled = !!this._todBusy;
  }

  async toggleTimeOfDay() {
    if (this._todBusy) return;
    const next = this.timeOfDay === "day" ? "night" : "day";
    await this.setTimeOfDay(next);
  }

  async setTimeOfDay(mode) {
    if (this._todBusy || (mode !== "day" && mode !== "night")) return;
    if (mode === this.timeOfDay) return;
    this._todBusy = true;
    this.syncTodButton();
    try {
      if (mode === "night") await this.transitionDayToNight();
      else await this.transitionNightToDay();
      this.timeOfDay = mode;
      document.body.classList.toggle("tod-night", mode === "night");
    } finally {
      this._todBusy = false;
      this.syncTodButton();
    }
  }

  easeInOut(t) {
    return t < 0.5 ? 2 * t * t : 1 - ((-2 * t + 2) ** 2) / 2;
  }

  easeOutCubic(t) {
    return 1 - (1 - t) ** 3;
  }

  easeOutBack(t) {
    const c = 1.70158;
    const t1 = t - 1;
    return 1 + c * t1 * t1 * t1 + t1 * t1;
  }

  animateValue(durationMs, onUpdate) {
    return new Promise((resolve) => {
      const t0 = performance.now();
      const step = (now) => {
        const u = Math.min(1, (now - t0) / durationMs);
        onUpdate(u);
        if (u < 1) requestAnimationFrame(step);
        else resolve();
      };
      requestAnimationFrame(step);
    });
  }

  sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  async transitionDayToNight() {
    const bgFrom = this.scene.background.clone();
    const expFrom = this.renderer.toneMappingExposure;

    // 1) Whole page fades to black; day lights die out.
    await this.animateValue(1400, (u) => {
      const e = this.easeInOut(u);
      this.setBlackout(e);
      this.setDayLightsLevel(1 - e);
      this.scene.background.copy(bgFrom).lerp(this._bgNight, e);
      this.renderer.toneMappingExposure = expFrom + (0.25 - expFrom) * e;
    });

    this.dayLightGroup.visible = false;
    this.setDayLightsLevel(0);
    this.beamGroup.visible = true;
    this.setNightFillLevel(0);
    this.setRobotLightsLevel(0);
    this.scene.background.copy(this._bgNight);

    // 2) Reveal the black stage.
    await this.animateValue(350, (u) => {
      this.setBlackout(1 - this.easeOutCubic(u));
    });
    this.setBlackout(0);
    await this.sleep(180);

    // 3) Three soft circular spots open on the mecha.
    await this.animateValue(520, (u) => {
      const e = this.easeOutCubic(u);
      this.setRobotLightsLevel(e);
      this.setNightFillLevel(0.25 + 0.75 * e);
      this.renderer.toneMappingExposure = 0.8 + 0.25 * e;
    });
    this.setRobotLightsLevel(1);
    this.setNightFillLevel(1);
    this.renderer.toneMappingExposure = 1.05;
  }

  async transitionNightToDay() {
    await this.animateValue(500, (u) => {
      const e = this.easeInOut(u);
      this.setRobotLightsLevel(1 - e);
      this.setNightFillLevel(1 - e);
      this.setBlackout(e * 0.55);
    });
    this.beamGroup.visible = false;
    this.dayLightGroup.visible = true;
    this.setDayLightsLevel(0);

    const bgFrom = this._bgNight.clone();
    await this.animateValue(900, (u) => {
      const e = this.easeInOut(u);
      this.setDayLightsLevel(e);
      this.scene.background.copy(bgFrom).lerp(this._bgDay, e);
      this.renderer.toneMappingExposure = 1.05 + (1.18 - 1.05) * e;
      this.setBlackout(0.55 * (1 - e));
    });
    this.applyDayLookImmediate();
  }

  createSpotlight({
    color,
    intensity,
    position,
    target,
    distance = 12,
    angle = Math.PI / 6,
    penumbra = 0.4,
    castShadow = false,
  }) {
    const spot = new THREE.SpotLight(color, intensity, distance, angle, penumbra, 1.55);
    spot.position.set(position[0], position[1], position[2]);
    spot.target.position.copy(target);
    spot.castShadow = castShadow;
    if (castShadow) {
      spot.shadow.mapSize.setScalar(2048);
      spot.shadow.bias = -0.0002;
      spot.shadow.normalBias = 0.02;
      spot.shadow.camera.near = 0.5;
      spot.shadow.camera.far = distance;
    }
    this.scene.add(spot);
    this.scene.add(spot.target);
    return spot;
  }

  /** Soft radial falloff texture for volumetric cones. */
  createBeamTexture() {
    if (this._beamTexture) return this._beamTexture;
    const size = 256;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    const g = ctx.createLinearGradient(0, 0, 0, size);
    g.addColorStop(0, "rgba(255,255,255,0.95)");
    g.addColorStop(0.35, "rgba(255,255,255,0.45)");
    g.addColorStop(1, "rgba(255,255,255,0.0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);

    // Soft radial edge so the cone looks round, not hard-sided.
    const img = ctx.getImageData(0, 0, size, size);
    const cx = size / 2;
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const i = (y * size + x) * 4;
        const nx = (x - cx) / cx;
        const edge = Math.max(0, 1 - nx * nx);
        img.data[i + 3] = Math.round(img.data[i + 3] * edge * edge);
      }
    }
    ctx.putImageData(img, 0, 0);
    this._beamTexture = new THREE.CanvasTexture(canvas);
    this._beamTexture.needsUpdate = true;
    return this._beamTexture;
  }

  createCircularBeam({ color, from, to, angle, opacity = 0.2 }) {
    const dir = new THREE.Vector3().subVectors(to, from);
    const length = Math.max(0.2, dir.length());
    // Apex at light source (local -Y), diverges to a disc at the target (local +Y).
    const radius = Math.tan(angle) * length;
    const geo = new THREE.CylinderGeometry(radius, 0.015, length, 48, 1, true);
    const mat = new THREE.MeshBasicMaterial({
      map: this.createBeamTexture(),
      color,
      transparent: true,
      opacity,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = 2;
    mesh.userData.baseOpacity = opacity;
    this.orientBeam(mesh, from, to);

    // Soft circular pool where the beam lands.
    const pool = new THREE.Mesh(
      new THREE.CircleGeometry(radius * 1.05, 48),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: opacity * 0.85,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
      })
    );
    pool.userData.baseOpacity = opacity * 0.85;
    pool.position.copy(to);
    pool.lookAt(from);
    pool.renderOrder = 2;

    // Tiny lamp head at source
    const bulb = new THREE.Mesh(
      new THREE.SphereGeometry(0.06, 16, 16),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.95,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    bulb.userData.baseOpacity = 0.95;
    bulb.position.copy(from);
    const group = new THREE.Group();
    group.add(mesh);
    group.add(pool);
    group.add(bulb);
    group.userData.beamMesh = mesh;
    group.userData.bulb = bulb;
    group.userData.pool = pool;
    return group;
  }

  orientBeam(mesh, from, to) {
    const mid = new THREE.Vector3().addVectors(from, to).multiplyScalar(0.5);
    const dir = new THREE.Vector3().subVectors(to, from).normalize();
    mesh.position.copy(mid);
    // Local +Y points toward the target → wide end of the cylinder lands as a disc.
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  }

  addGround() {
    // Shrink by 1/3 → keep 2/3 of original size.
    const s = 2 / 3;
    const discR = 5.5 * s;
    const apronOuter = 7.2 * s;

    const groundMat = new THREE.MeshPhysicalMaterial({
      color: 0xe8e4dc,
      roughness: 0.72,
      metalness: 0.08,
      clearcoat: 0.2,
      clearcoatRoughness: 0.4,
      side: THREE.DoubleSide,
    });
    const ground = new THREE.Mesh(new THREE.CircleGeometry(discR, 96), groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.002;
    ground.receiveShadow = true;
    ground.name = "Ground";
    this.scene.add(ground);

    // Soft outer ring so the disc doesn't float over empty void.
    const apron = new THREE.Mesh(
      new THREE.RingGeometry(discR, apronOuter, 96),
      new THREE.MeshPhysicalMaterial({
        color: 0xc8b89a,
        roughness: 0.88,
        metalness: 0.05,
        side: THREE.DoubleSide,
      })
    );
    apron.rotation.x = -Math.PI / 2;
    apron.position.y = -0.003;
    apron.receiveShadow = true;
    this.scene.add(apron);

  }

  /**
   * Apply MuJoCo scene props from websocket snapshot.
   * Robot URDF uses rotation.x = -PI/2 (Z-up → Y-up); sword props live in the same frame.
   * Red handle marker is parented to the sword body frame so it stays glued to the grip
   * under pose edits. When attached, cosmetics are paused so the
   * mesh matches MuJoCo kinematic stick.
   */
  applySceneConfig(sceneCfg) {
    if (!sceneCfg || typeof sceneCfg !== "object") return;
    const name = sceneCfg.name || "free_space";
    if (name === "table_place") {
      this.clearReachSwordProps();
      this.clearShieldProps();
      this.clearExecutionPointMarker();
      this._reachSwordKey = null;
      this._reachShieldKey = null;
      this._syncTablePlaceScene(sceneCfg);
      this.setGraspAttached({
        left: false,
        right: Boolean(sceneCfg.right_attached || sceneCfg.peg?.attached),
      });
      this._refreshGraspFx?.();
      return;
    }
    if (name !== "reach_sword") {
      this.clearReachSwordProps();
      this.clearShieldProps();
      this.clearExecutionPointMarker();
      this.clearTablePlaceProps();
      this._reachSwordKey = null;
      this._reachShieldKey = null;
      this.setGraspAttached({ left: false, right: false });
      return;
    }
    this.clearTablePlaceProps();
    this._syncExecutionPoint(sceneCfg);
    const sword = sceneCfg.sword;
    if (!sword || sword.present === false || !sword.url) {
      this.clearReachSwordProps();
      this._reachSwordKey = null;
      this.setGraspAttached({
        left: Boolean(sceneCfg.left_attached || sceneCfg.shield?.attached),
        right: false,
      });
      if (Array.isArray(sceneCfg.right_tcp_local) && sceneCfg.right_tcp_local.length >= 3) {
        this._rightTcpLocal = sceneCfg.right_tcp_local.map(Number);
      }
      if (Array.isArray(sceneCfg.left_tcp_local) && sceneCfg.left_tcp_local.length >= 3) {
        this._leftTcpLocal = sceneCfg.left_tcp_local.map(Number);
      }
      this._syncShieldScene(sceneCfg);
      this._refreshGraspFx();
      return;
    }

    const [px, py, pz] = sword.position || [0, 0, 0];
    const [qw, qx, qy, qz] = sword.quaternion_wxyz || [1, 0, 0, 0];
    const scale = Number(sword.scale) || 1;
    const swordColor = Number.isFinite(Number(sword.color)) ? Number(sword.color) : 0xf5f7fa;
    const [hx, hy, hz] = sword.handle_local || [0, 0, 0];
    if (Array.isArray(sceneCfg.right_tcp_local) && sceneCfg.right_tcp_local.length >= 3) {
      this._rightTcpLocal = sceneCfg.right_tcp_local.map(Number);
    }
    if (Array.isArray(sceneCfg.left_tcp_local) && sceneCfg.left_tcp_local.length >= 3) {
      this._leftTcpLocal = sceneCfg.left_tcp_local.map(Number);
    }
    this.setGraspAttached({
      left: Boolean(sceneCfg.left_attached || sceneCfg.shield?.attached),
      right: Boolean(sceneCfg.right_attached || sceneCfg.attached || sword.attached),
    });

    const applySwordMaterial = (mat, colorHex) => {
      if (!mat) return;
      mat.color?.setHex(colorHex);
      mat.metalness = 0.18;
      mat.roughness = 0.42;
      mat.clearcoat = 0.35;
      mat.clearcoatRoughness = 0.35;
      if (mat.emissive) mat.emissive.setHex(0x000000);
      mat.emissiveIntensity = 0;
      mat.needsUpdate = true;
    };

    // Always keep animation base in sync (even when config key is unchanged).
    this._swordBasePos = [px, py, pz];
    if (!this._swordBaseQuat) this._swordBaseQuat = new THREE.Quaternion();
    this._swordBaseQuat.set(qx, qy, qz, qw);
    this._swordHandleLocal = [hx, hy, hz];

    // Exclude live pose from rebuild key so attach/carry does not rebuild the mesh.
    const key = JSON.stringify({
      url: sword.url,
      scale: sword.scale,
      handle_local: sword.handle_local,
      color: sword.color,
    });
    // Already loaded for this config — do not rebuild (was causing flicker every WS frame).
    if (this._reachSwordKey === key && this._reachSwordGroup && this._swordPivot) {
      if (this._swordHandleMarker) {
        this._swordHandleMarker.position.set(hx, hy, hz);
      }
      // Pose-only updates (e.g. Reach-IK replay handle) must still move the pivot.
      if (!this._swordAttached) {
        this._swordPivot.position.set(px, py, pz);
        if (this._swordOriented) this._swordOriented.quaternion.copy(this._swordBaseQuat);
      }
      this._syncShieldScene(sceneCfg);
      this._refreshGraspFx();
      return;
    }

    // Pose/color-only change: update materials + local marker; keep pivot for FX.
    if (
      this._swordMesh &&
      this._swordOriented &&
      this._swordPivot &&
      this._reachSwordGroup &&
      this._reachSwordUrl === sword.url &&
      Number(this._reachSwordScale) === scale
    ) {
      applySwordMaterial(this._swordMesh.material, swordColor);
      if (this._swordHandleMarker) {
        this._swordHandleMarker.position.set(hx, hy, hz);
      }
      if (!this._swordAttached) {
        this._swordPivot.position.set(px, py, pz);
        this._swordOriented.quaternion.copy(this._swordBaseQuat);
      }
      this._reachSwordKey = key;
      this._syncShieldScene(sceneCfg);
      this._refreshGraspFx();
      return;
    }

    this.clearReachSwordProps();
    this._reachSwordKey = key;
    this._reachSwordUrl = sword.url;
    this._reachSwordScale = scale;
    this._swordSpinAngle = 0;
    this._swordBasePos = [px, py, pz];
    this._swordBaseQuat = new THREE.Quaternion(qx, qy, qz, qw);
    this._swordHandleLocal = [hx, hy, hz];

    if (!this._mujocoWorld) {
      this._mujocoWorld = new THREE.Group();
      this._mujocoWorld.name = "MujocoWorld";
      this._mujocoWorld.rotation.x = -Math.PI / 2;
      this.scene.add(this._mujocoWorld);
    }
    if (this.robot) {
      this._mujocoWorld.position.y = this.robot.position.y;
    }

    const group = new THREE.Group();
    group.name = "ReachSwordProps";
    this._reachSwordGroup = group;
    this._mujocoWorld.add(group);

    // pivot (world pos) → oriented (body quat) → mesh + handle marker (local).
    const pivot = new THREE.Group();
    pivot.name = "SwordPivot";
    pivot.position.set(px, py, pz);
    group.add(pivot);
    this._swordPivot = pivot;

    const oriented = new THREE.Group();
    oriented.name = "SwordOriented";
    oriented.quaternion.copy(this._swordBaseQuat);
    pivot.add(oriented);
    this._swordOriented = oriented;

    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.028, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xe23b2c, depthTest: true })
    );
    marker.name = "SwordHandleMarker";
    marker.position.set(hx, hy, hz);
    oriented.add(marker);
    this._swordHandleMarker = marker;

    const loadId = key;
    const loader = new STLLoader();
    loader.load(
      sword.url,
      (geometry) => {
        // Drop stale async results if config changed while loading.
        if (this._reachSwordKey !== loadId || this._reachSwordGroup !== group || this._swordOriented !== oriented) {
          geometry.dispose();
          return;
        }
        geometry.computeVertexNormals();
        const material = new THREE.MeshPhysicalMaterial({
          color: swordColor,
          metalness: 0.18,
          roughness: 0.42,
          clearcoat: 0.35,
          clearcoatRoughness: 0.35,
          emissive: 0x000000,
          emissiveIntensity: 0,
          side: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = "Sword";
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.scale.setScalar(scale);
        mesh.position.set(0, 0, 0);
        mesh.quaternion.identity();
        oriented.add(mesh);
        this._swordMesh = mesh;
        this._refreshGraspFx();
      },
      undefined,
      (err) => {
        console.error("sword.stl 加载失败", err);
        this.onSystemMessage(`剑模型加载失败: ${err?.message || err}`);
      }
    );
    this._syncShieldScene(sceneCfg);
    this._refreshGraspFx();
  }

  /** Sync left-arm shield/helmet prop from snapshot (kinematic, opposite side of sword). */
  _syncShieldScene(sceneCfg) {
    const shield = sceneCfg?.shield;
    if (!shield || shield.present === false || !shield.url) {
      this.clearShieldProps();
      this._reachShieldKey = null;
      return;
    }
    const [px, py, pz] = shield.position || [0, 0, 0];
    const [qw, qx, qy, qz] = shield.quaternion_wxyz || [1, 0, 0, 0];
    const scale = Number(shield.scale) || 1;
    const color = Number.isFinite(Number(shield.color)) ? Number(shield.color) : 0xb8c6db;
    const [hx, hy, hz] = shield.handle_local || [0, 0, 0];

    this._shieldBasePos = [px, py, pz];
    if (!this._shieldBaseQuat) this._shieldBaseQuat = new THREE.Quaternion();
    this._shieldBaseQuat.set(qx, qy, qz, qw);
    this._shieldHandleLocal = [hx, hy, hz];

    const key = JSON.stringify({
      url: shield.url,
      scale: shield.scale,
      handle_local: shield.handle_local,
      color: shield.color,
    });
    if (this._reachShieldKey === key && this._shieldPivot && this._shieldOriented) {
      if (this._shieldHandleMarker) this._shieldHandleMarker.position.set(hx, hy, hz);
      return;
    }
    if (
      this._shieldMesh &&
      this._shieldOriented &&
      this._shieldPivot &&
      this._reachShieldUrl === shield.url &&
      Number(this._reachShieldScale) === scale
    ) {
      if (this._shieldMesh.material?.color) this._shieldMesh.material.color.setHex(color);
      if (this._shieldHandleMarker) this._shieldHandleMarker.position.set(hx, hy, hz);
      this._reachShieldKey = key;
      return;
    }

    this.clearShieldProps();
    this._reachShieldKey = key;
    this._reachShieldUrl = shield.url;
    this._reachShieldScale = scale;
    this._shieldBasePos = [px, py, pz];
    this._shieldBaseQuat = new THREE.Quaternion(qx, qy, qz, qw);
    this._shieldHandleLocal = [hx, hy, hz];

    if (!this._mujocoWorld) {
      this._mujocoWorld = new THREE.Group();
      this._mujocoWorld.name = "MujocoWorld";
      this._mujocoWorld.rotation.x = -Math.PI / 2;
      this.scene.add(this._mujocoWorld);
    }
    if (this.robot) this._mujocoWorld.position.y = this.robot.position.y;

    const group = new THREE.Group();
    group.name = "ReachShieldProps";
    this._reachShieldGroup = group;
    this._mujocoWorld.add(group);

    const pivot = new THREE.Group();
    pivot.name = "ShieldPivot";
    pivot.position.set(px, py, pz);
    group.add(pivot);
    this._shieldPivot = pivot;

    const oriented = new THREE.Group();
    oriented.name = "ShieldOriented";
    oriented.quaternion.copy(this._shieldBaseQuat);
    pivot.add(oriented);
    this._shieldOriented = oriented;

    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xf2281a })
    );
    marker.name = "ShieldHandleMarker";
    marker.position.set(hx, hy, hz);
    oriented.add(marker);
    this._shieldHandleMarker = marker;

    const loadId = key;
    const loader = new STLLoader();
    loader.load(
      shield.url,
      (geometry) => {
        if (this._reachShieldKey !== loadId || this._reachShieldGroup !== group || this._shieldOriented !== oriented) {
          geometry.dispose();
          return;
        }
        geometry.computeVertexNormals();
        const material = new THREE.MeshPhysicalMaterial({
          color,
          metalness: 0.22,
          roughness: 0.48,
          clearcoat: 0.25,
          clearcoatRoughness: 0.4,
          emissive: 0x000000,
          emissiveIntensity: 0,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.name = "Shield";
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.scale.setScalar(scale);
        oriented.add(mesh);
        this._shieldMesh = mesh;
        this._refreshGraspFx();
      },
      undefined,
      (err) => {
        console.error("helmet.stl 加载失败", err);
        this.onSystemMessage(`盾牌模型加载失败: ${err?.message || err}`);
      }
    );
  }

  /** Pin shield to configured pose (no bob). When attached, TCP follow owns the pose. */
  updateShieldVisualFx(_deltaSec) {
    if (!this._shieldPivot || !this._shieldOriented || !this._shieldBasePos || !this._shieldBaseQuat) {
      return;
    }
    if (this._shieldAttached) return;
    const [bx, by, bz] = this._shieldBasePos;
    this._shieldPivot.position.set(bx, by, bz);
    this._shieldOriented.quaternion.copy(this._shieldBaseQuat);
  }

  /** Pin sword to configured pose (no spin/bob). When attached, TCP follow owns the pose. */
  updateSwordVisualFx(_deltaSec) {
    if (!this._swordPivot || !this._swordOriented || !this._swordBasePos || !this._swordBaseQuat) return;
    if (this._swordAttached) {
      return;
    }
    this._swordSpinAngle = 0;
    const [bx, by, bz] = this._swordBasePos;
    this._swordPivot.position.set(bx, by, bz);
    this._swordOriented.quaternion.copy(this._swordBaseQuat);
  }

  setGraspAttached({ left = false, right = false } = {}) {
    const nextLeft = Boolean(left);
    const nextRight = Boolean(right);
    if (nextRight !== this._swordAttached) {
      this._swordAttached = nextRight;
      if (nextRight) {
        this._captureAttachRel("right");
        this._updateAttachedFollow("right");
      } else {
        this._bakeSwordWorldToBase();
        this._swordAttachRelQuat = null;
      }
    } else if (nextRight && !this._swordAttachRelQuat) {
      this._captureAttachRel("right");
    }
    if (nextLeft !== this._shieldAttached) {
      this._shieldAttached = nextLeft;
      if (nextLeft) {
        this._captureAttachRel("left");
        this._updateAttachedFollow("left");
      } else {
        this._bakeShieldWorldToBase();
        this._shieldAttachRelQuat = null;
      }
    } else if (nextLeft && !this._shieldAttachRelQuat) {
      this._captureAttachRel("left");
    }
    this._refreshGraspFx();
  }

  /** @deprecated use setGraspAttached */
  setSwordAttached(active) {
    this.setGraspAttached({ left: this._shieldAttached, right: Boolean(active) });
  }

  _ensureTcpHelper(side) {
    const isLeft = side === "left";
    const key = isLeft ? "_leftTcpHelper" : "_rightTcpHelper";
    if (this[key]?.parent) return this[key];
    if (!this.robot) return null;
    const linkName = isLeft ? "Left_Link7" : "Right_Link7";
    const link =
      this.robot.links?.[linkName] ||
      this.robot.frames?.[linkName] ||
      this.robot.getObjectByName?.(linkName);
    if (!link) return null;
    if (this[key]) this[key].removeFromParent?.();
    const helper = new THREE.Object3D();
    helper.name = isLeft ? "LeftTcpHelper" : "RightTcpHelper";
    const local = (isLeft ? this._leftTcpLocal : this._rightTcpLocal) || [0.26, 0, 0];
    helper.position.set(Number(local[0]) || 0, Number(local[1]) || 0, Number(local[2]) || 0);
    link.add(helper);
    this[key] = helper;
    return helper;
  }

  _captureAttachRel(side) {
    const isLeft = side === "left";
    const tcp = this._ensureTcpHelper(side);
    const baseQuat = isLeft ? this._shieldBaseQuat : this._swordBaseQuat;
    const relKey = isLeft ? "_shieldAttachRelQuat" : "_swordAttachRelQuat";
    if (!tcp || !baseQuat || !this._mujocoWorld) return;
    this.robot?.updateMatrixWorld(true);
    this._mujocoWorld.updateMatrixWorld(true);
    if (!this._tmpTcpQ) this._tmpTcpQ = new THREE.Quaternion();
    if (!this._tmpSwordQ) this._tmpSwordQ = new THREE.Quaternion();
    if (!this._tmpWorldQ) this._tmpWorldQ = new THREE.Quaternion();
    tcp.getWorldQuaternion(this._tmpTcpQ);
    this._mujocoWorld.getWorldQuaternion(this._tmpWorldQ);
    this._tmpSwordQ.copy(this._tmpWorldQ).multiply(baseQuat);
    this[relKey] = this._tmpTcpQ.clone().invert().multiply(this._tmpSwordQ);
  }

  _bakeSwordWorldToBase() {
    if (!this._swordPivot || !this._swordOriented) return;
    this._swordBasePos = [
      this._swordPivot.position.x,
      this._swordPivot.position.y,
      this._swordPivot.position.z,
    ];
    if (!this._swordBaseQuat) this._swordBaseQuat = new THREE.Quaternion();
    this._swordBaseQuat.copy(this._swordOriented.quaternion);
  }

  _bakeShieldWorldToBase() {
    if (!this._shieldPivot || !this._shieldOriented) return;
    this._shieldBasePos = [
      this._shieldPivot.position.x,
      this._shieldPivot.position.y,
      this._shieldPivot.position.z,
    ];
    if (!this._shieldBaseQuat) this._shieldBaseQuat = new THREE.Quaternion();
    this._shieldBaseQuat.copy(this._shieldOriented.quaternion);
  }

  _updateAttachedFollow(side) {
    const isLeft = side === "left";
    const attached = isLeft ? this._shieldAttached : this._swordAttached;
    const pivot = isLeft ? this._shieldPivot : this._swordPivot;
    const oriented = isLeft ? this._shieldOriented : this._swordOriented;
    const handleLocal = isLeft ? this._shieldHandleLocal : this._swordHandleLocal;
    const relQuat = isLeft ? this._shieldAttachRelQuat : this._swordAttachRelQuat;
    if (!attached || !pivot || !oriented || !this._mujocoWorld) return;
    if (!relQuat) this._captureAttachRel(side);
    const tcp = this._ensureTcpHelper(side);
    const rel = isLeft ? this._shieldAttachRelQuat : this._swordAttachRelQuat;
    if (!tcp || !rel) return;

    if (!this._tmpTcpPos) this._tmpTcpPos = new THREE.Vector3();
    if (!this._tmpTcpQ) this._tmpTcpQ = new THREE.Quaternion();
    if (!this._tmpSwordQ) this._tmpSwordQ = new THREE.Quaternion();
    if (!this._tmpOffset) this._tmpOffset = new THREE.Vector3();
    if (!this._tmpBodyWorld) this._tmpBodyWorld = new THREE.Vector3();
    if (!this._tmpWorldQ) this._tmpWorldQ = new THREE.Quaternion();
    if (!this._tmpInvMat) this._tmpInvMat = new THREE.Matrix4();

    tcp.getWorldPosition(this._tmpTcpPos);
    tcp.getWorldQuaternion(this._tmpTcpQ);
    this._tmpSwordQ.copy(this._tmpTcpQ).multiply(rel);
    const [hx, hy, hz] = handleLocal || [0, 0, 0];
    this._tmpOffset.set(hx, hy, hz).applyQuaternion(this._tmpSwordQ);
    this._tmpBodyWorld.copy(this._tmpTcpPos).sub(this._tmpOffset);
    this._tmpInvMat.copy(this._mujocoWorld.matrixWorld).invert();
    this._tmpBodyWorld.applyMatrix4(this._tmpInvMat);
    this._mujocoWorld.getWorldQuaternion(this._tmpWorldQ);
    const localQ = this._tmpWorldQ.clone().invert().multiply(this._tmpSwordQ);
    pivot.position.copy(this._tmpBodyWorld);
    oriented.quaternion.copy(localQ);
    if (isLeft) {
      this._shieldBasePos = [this._tmpBodyWorld.x, this._tmpBodyWorld.y, this._tmpBodyWorld.z];
      if (!this._shieldBaseQuat) this._shieldBaseQuat = new THREE.Quaternion();
      this._shieldBaseQuat.copy(localQ);
    } else {
      this._swordBasePos = [this._tmpBodyWorld.x, this._tmpBodyWorld.y, this._tmpBodyWorld.z];
      if (!this._swordBaseQuat) this._swordBaseQuat = new THREE.Quaternion();
      this._swordBaseQuat.copy(localQ);
    }
  }

  _updateAttachedSwordFollow() {
    this._updateAttachedFollow("right");
    this._updateAttachedFollow("left");
  }

  clearShieldProps() {
    if (this._reachShieldGroup) {
      this._reachShieldGroup.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose?.();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose?.());
          else obj.material.dispose?.();
        }
      });
      if (this._reachShieldGroup.parent) {
        this._reachShieldGroup.parent.remove(this._reachShieldGroup);
      }
    }
    this._reachShieldGroup = null;
    this._shieldPivot = null;
    this._shieldOriented = null;
    this._shieldMesh = null;
    this._shieldHandleMarker = null;
    this._shieldBasePos = null;
    this._shieldBaseQuat = null;
    this._shieldHandleLocal = null;
    this._shieldAttached = false;
    this._shieldAttachRelQuat = null;
    this._reachShieldKey = null;
  }

  clearReachSwordProps() {
    if (this._reachSwordGroup) {
      this._reachSwordGroup.traverse((obj) => {
        if (obj.geometry) obj.geometry.dispose?.();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose?.());
          else obj.material.dispose?.();
        }
      });
      if (this._reachSwordGroup.parent) {
        this._reachSwordGroup.parent.remove(this._reachSwordGroup);
      }
    }
    this._reachSwordGroup = null;
    this._swordPivot = null;
    this._swordOriented = null;
    this._swordMesh = null;
    this._swordHandleMarker = null;
    this._swordBasePos = null;
    this._swordBaseQuat = null;
    this._swordHandleLocal = null;
    this._swordSpinAngle = 0;
    this._swordAttached = false;
    this._swordAttachRelQuat = null;
    this._attachRelQuat = null;
  }

  clearTablePlaceProps() {
    if (this._tableMesh) {
      this._tableMesh.parent?.remove(this._tableMesh);
      this._tableMesh.geometry?.dispose?.();
      this._tableMesh.material?.dispose?.();
      this._tableMesh = null;
    }
    if (this._pegGroup) {
      this._pegGroup.parent?.remove(this._pegGroup);
      this._pegGroup.traverse((obj) => {
        obj.geometry?.dispose?.();
        if (obj.material) {
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose?.());
          else obj.material.dispose?.();
        }
      });
      this._pegGroup = null;
    }
    if (this._circleMesh) {
      this._circleMesh.parent?.remove(this._circleMesh);
      this._circleMesh.geometry?.dispose?.();
      this._circleMesh.material?.dispose?.();
      this._circleMesh = null;
    }
  }

  _syncTablePlaceScene(sceneCfg) {
    if (!this._mujocoWorld) {
      this._mujocoWorld = new THREE.Group();
      this._mujocoWorld.name = "MujocoWorld";
      // Match URDF Z-up → Three.js Y-up (same as sword props).
      this._mujocoWorld.rotation.x = -Math.PI / 2;
      this.scene.add(this._mujocoWorld);
    }

    const table = sceneCfg.table;
    if (table?.present !== false && Array.isArray(table?.position) && Array.isArray(table?.half_size)) {
      const [px, py, pz] = table.position.map(Number);
      const [hx, hy, hz] = table.half_size.map(Number);
      const color = Number.isFinite(Number(table.color)) ? Number(table.color) : 0x8c6b47;
      if (!this._tableMesh) {
        const geo = new THREE.BoxGeometry(hx * 2, hy * 2, hz * 2);
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0.05 });
        this._tableMesh = new THREE.Mesh(geo, mat);
        this._tableMesh.name = "TablePlaceTable";
        this._mujocoWorld.add(this._tableMesh);
      } else {
        this._tableMesh.material.color.setHex(color);
      }
      this._tableMesh.position.set(px, py, pz);
      this._tableMesh.scale.set(1, 1, 1);
    }

    const peg = sceneCfg.peg;
    if (peg?.present !== false && Array.isArray(peg?.position)) {
      const [px, py, pz] = peg.position.map(Number);
      const [qw, qx, qy, qz] = (peg.quaternion_wxyz || [1, 0, 0, 0]).map(Number);
      const radius = Number(peg.radius) || 0.018;
      const halfH = Number(peg.half_height) || 0.045;
      const color = Number.isFinite(Number(peg.color)) ? Number(peg.color) : 0xd95a33;
      if (!this._pegGroup) {
        this._pegGroup = new THREE.Group();
        this._pegGroup.name = "TablePlacePeg";
        const geo = new THREE.CylinderGeometry(radius, radius, halfH * 2, 24);
        // CylinderGeometry is Y-up; MuJoCo cylinder is Z-up → rotate.
        geo.rotateX(Math.PI / 2);
        const mat = new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.1 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.name = "PegMesh";
        this._pegGroup.add(mesh);
        this._mujocoWorld.add(this._pegGroup);
      } else {
        const mesh = this._pegGroup.getObjectByName("PegMesh");
        if (mesh?.material) mesh.material.color.setHex(color);
      }
      this._pegGroup.position.set(px, py, pz);
      this._pegGroup.quaternion.set(qx, qy, qz, qw);
    }

    const circle = sceneCfg.circle;
    if (circle?.present !== false && Array.isArray(circle?.center)) {
      const [cx, cy, cz] = circle.center.map(Number);
      const radius = Number(circle.radius) || 0.06;
      const color = Number.isFinite(Number(circle.color)) ? Number(circle.color) : 0x26bf59;
      if (!this._circleMesh) {
        const geo = new THREE.RingGeometry(radius * 0.72, radius, 48);
        const mat = new THREE.MeshBasicMaterial({
          color,
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
        });
        this._circleMesh = new THREE.Mesh(geo, mat);
        this._circleMesh.name = "TablePlaceCircle";
        this._mujocoWorld.add(this._circleMesh);
      } else {
        this._circleMesh.material.color.setHex(color);
      }
      // Ring lies in XY of local frame; table is Z-up in MuJoCo world group.
      this._circleMesh.position.set(cx, cy, cz + 0.002);
      this._circleMesh.rotation.set(0, 0, 0);
      this._circleMesh.scale.set(1, 1, 1);
    }
  }

  /** Green post-grasp waypoint marker in MuJoCo/world frame (robot base). */
  _syncExecutionPoint(sceneCfg) {
    const ep = sceneCfg?.execution_point;
    if (!ep || !ep.enabled) {
      this.clearExecutionPointMarker();
      return;
    }
    const pos = Array.isArray(ep.position) ? ep.position : null;
    if (!pos || pos.length < 3) {
      this.clearExecutionPointMarker();
      return;
    }
    const [x, y, z] = pos.map(Number);
    if (![x, y, z].every(Number.isFinite)) {
      this.clearExecutionPointMarker();
      return;
    }
    const color = Number.isFinite(Number(ep.color)) ? Number(ep.color) : 0x2ecc71;
    const passed = Boolean(ep.passed);

    if (!this._mujocoWorld) {
      this._mujocoWorld = new THREE.Group();
      this._mujocoWorld.name = "MujocoWorld";
      this._mujocoWorld.rotation.x = -Math.PI / 2;
      this.scene.add(this._mujocoWorld);
    }
    if (this.robot) {
      this._mujocoWorld.position.y = this.robot.position.y;
    }

    if (!this._executionPointMarker) {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.032, 16, 16),
        new THREE.MeshBasicMaterial({ color, depthTest: true })
      );
      mesh.name = "ExecutionPointMarker";
      this._mujocoWorld.add(mesh);
      this._executionPointMarker = mesh;
    }
    this._executionPointMarker.position.set(x, y, z);
    if (this._executionPointMarker.material?.color) {
      this._executionPointMarker.material.color.setHex(passed ? 0xa8e6a1 : color);
    }
    this._executionPointMarker.visible = true;
  }

  clearExecutionPointMarker() {
    if (this._executionPointMarker) {
      const mat = this._executionPointMarker.material;
      const geo = this._executionPointMarker.geometry;
      if (this._executionPointMarker.parent) {
        this._executionPointMarker.parent.remove(this._executionPointMarker);
      }
      geo?.dispose?.();
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose?.());
      else mat?.dispose?.();
    }
    this._executionPointMarker = null;
  }

  async loadRobot() {
    const urdfUrl = "/robot/model.urdf?v=mecha8";
    const sampleMesh = "/urdf/meshes/meshes__mecha_body.stl?v=mecha8";
    try {
      const urdfCheck = await fetch(urdfUrl);
      if (!urdfCheck.ok) throw new Error(`URDF HTTP ${urdfCheck.status}`);
      const meshCheck = await fetch(sampleMesh);
      if (!meshCheck.ok) throw new Error(`mesh HTTP ${meshCheck.status}`);
    } catch (err) {
      console.error("静态资源预检失败", err);
      this.loadState.querySelector("span:last-child").textContent = "URDF/网格无法访问";
      this.modelState.textContent = err.message;
      this.onSystemMessage(`静态资源访问失败: ${err.message}`);
      return;
    }

    const manager = new THREE.LoadingManager();
    const loader = new URDFLoader(manager);
    const meshErrors = [];

    manager.onProgress = (_url, loaded, total) => {
      this.loadState.querySelector("span:last-child").textContent =
        `正在加载机器人模型 ${loaded}/${total}`;
    };

    manager.onError = (url) => {
      const text = `网格加载失败: ${url}`;
      meshErrors.push(text);
      console.error(text);
    };

    const alignAndFit = () => {
      if (!this.robot) return;
      this.robot.updateMatrixWorld(true);
      const box0 = new THREE.Box3().setFromObject(this.robot);
      const size0 = box0.getSize(new THREE.Vector3());
      if (size0.y < size0.z * 0.7) {
        this.robot.rotation.x = -Math.PI / 2;
        this.robot.updateMatrixWorld(true);
      }
      const box = new THREE.Box3().setFromObject(this.robot);
      const bottomY = box.min.y;
      this.robot.position.y = -bottomY;
      this.robot.updateMatrixWorld(true);
      if (this._mujocoWorld) {
        this._mujocoWorld.position.y = this.robot.position.y;
      }
      this.fitView();
    };

    manager.onLoad = () => {
      alignAndFit();
      this.applyTechMaterials();
      this.loadState.style.display = "none";
      let visualCount = 0;
      this.robot.traverse((c) => { if (c.isMesh) visualCount += 1; });
      const jointCount = Object.keys(this.robot.joints || {}).length;
      this.modelState.textContent = `${jointCount} 个关节 / ${visualCount} 个网格`;
      if (visualCount === 0) {
        this.onSystemMessage("URDF 解析完成，但没有加载到可见网格，请检查网络请求里的 .STL 是否 404。");
      } else if (meshErrors.length) {
        this.onSystemMessage(`部分网格加载失败 (${meshErrors.length} 个)，模型可能不完整。`);
      }
    };

    loader.load(
      urdfUrl,
      (robot) => {
        this.robot = robot;
        robot.name = "RobotURDF";
        robot.rotation.set(0, 0, 0);
        robot.scale.setScalar(1);
        this.applyTechMaterials();

        this.scene.add(robot);

        const axes = new THREE.AxesHelper(0.3);
        axes.position.y = 0.02;
        this.scene.add(axes);
      },
      undefined,
      (error) => {
        console.error("URDF 加载失败", error);
        this.loadState.querySelector("span:last-child").textContent = "URDF 加载失败";
        this.modelState.textContent = error?.message || error;
        this.onSystemMessage(`URDF 加载失败: ${error?.message || error}`);
      }
    );
  }

  /** Bright Gundam-like white / platinum / gold palette (Web 3D only). */
  applyTechMaterials() {
    if (!this.robot) return;

    // RX-78 inspired: white armor, champagne platinum, gold trim, light blue/red accents.
    const palette = {
      base: { color: 0xf2f4f7, metalness: 0.28, roughness: 0.42 },       // armor white
      left: { color: 0xe8ecf0, metalness: 0.35, roughness: 0.36 },       // cool platinum
      right: { color: 0xeef1f4, metalness: 0.32, roughness: 0.38 },      // warm platinum
      gripper: { color: 0xd4af37, metalness: 0.78, roughness: 0.28 },    // gold
      camera: { color: 0xc5ccd4, metalness: 0.45, roughness: 0.4 },      // light gunmetal
      accent: { color: 0xc9a227, metalness: 0.82, roughness: 0.26 },     // richer gold
      markBlue: { color: 0x3a6ea5, metalness: 0.4, roughness: 0.4 },    // soft gundam blue
      markRed: { color: 0xb84a3c, metalness: 0.35, roughness: 0.42 },    // soft gundam red
    };

    const resolveRole = (name) => {
      const n = (name || "").toLowerCase();
      if (/camera|realsense/.test(n)) return "camera";
      if (/gripper|flange/.test(n)) return "gripper";
      if (/head|bracket/.test(n)) return "accent";
      if (/wrist.*mount|mount/.test(n)) return "markBlue";
      // Alternate a few distal links toward gold/blue for panel-break interest.
      if (/link7|j7_/.test(n)) return "accent";
      if (/link5|j5/.test(n)) return "markBlue";
      if (/link3|j3/.test(n)) return "left";
      if (/^left_|left_link|j\d*l|j0l|j1l|j2l|j3l|j4l|j5l|j6l|j7_left/.test(n)) return "left";
      if (/^right_|right_link|j\d*r|j0r|j1r|j2r|j3r|j4r|j5r|j6r|j7_right/.test(n)) return "right";
      if (/base|world|torso|j0/.test(n)) return "base";
      if (/left/.test(n)) return "left";
      if (/right/.test(n)) return "right";
      return "base";
    };

    const linkNameFor = (obj) => {
      let cur = obj;
      while (cur) {
        if (cur.isURDFLink && cur.name) return cur.name;
        if (cur.name && /link|base|gripper|camera|j\d|head|flange|realsense/i.test(cur.name)) {
          return cur.name;
        }
        cur = cur.parent;
      }
      return obj.name || "";
    };

    this.robot.traverse((child) => {
      child.castShadow = true;
      child.receiveShadow = true;
      // urdf-loader also builds meshes for <collision>; those still pointed at the
      // old torso STLs and were being drawn on top of / instead of mecha.
      if (child.isURDFCollider) {
        child.visible = false;
        return;
      }
      if (!child.isMesh || !child.material) return;

      const role = resolveRole(linkNameFor(child));
      const spec = palette[role] || palette.base;
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((mat) => {
        if (!mat) return;
        if (mat.color) mat.color.setHex(spec.color);
        if ("metalness" in mat) mat.metalness = spec.metalness;
        if ("roughness" in mat) mat.roughness = spec.roughness;
        if ("clearcoat" in mat) mat.clearcoat = role === "gripper" || role === "accent" ? 0.35 : 0.18;
        mat.side = THREE.DoubleSide;
        mat.needsUpdate = true;
      });
    });
  }

  applyJointState(jointState) {
    if (!this.robot || !jointState?.name || !jointState?.position) return;
    jointState.name.forEach((name, index) => {
      const value = Number(jointState.position[index]);
      if (Number.isFinite(value)) this.jointTargets.set(name, value);
    });
  }

  updateRenderedRobot(deltaSec) {
    if (!this.robot || this.jointTargets.size === 0) return;
    const carrying = Boolean(this._swordAttached || this._shieldAttached);
    const alpha = carrying ? 1 : 1 - Math.exp(-Math.max(0, deltaSec) / 0.035);
    let changed = false;
    for (const [name, target] of this.jointTargets) {
      const joint = this.robot.joints?.[name];
      if (!joint) continue;
      const current = this.jointDisplay.has(name) ? this.jointDisplay.get(name) : target;
      let delta = target - current;
      if (joint.jointType === "continuous") delta = Math.atan2(Math.sin(delta), Math.cos(delta));
      const next = Math.abs(delta) < 1e-5 || carrying ? target : current + delta * alpha;
      if (Math.abs(next - current) > 1e-7) changed = true;
      this.jointDisplay.set(name, next);
      joint.setJointValue(next);
    }
    if (changed || carrying) this.robot.updateMatrixWorld(true);
  }

  setDefaultView() {
    this.camera.position.set(2.8, 1.8, 3.6);
    this.controls.target.set(0, 1.0, 0);
    this.controls.update();
  }

  resetView() {
    this.setDefaultView();
  }

  frontView() {
    const t = this.controls.target.clone();
    this.camera.position.set(t.x, t.y + 0.15, t.z + 4.2);
    this.camera.up.set(0, 1, 0);
    this.controls.update();
  }

  fitView() {
    if (!this.robot) { this.setDefaultView(); return; }
    this.robot.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.robot);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const dist = maxDim / (2 * Math.tan((this.camera.fov * Math.PI) / 360));
    this.controls.target.copy(center);
    this.camera.position.set(center.x + dist * 0.6, center.y + dist * 0.25, center.z + dist * 1.05);
    this.camera.near = Math.max(0.001, dist / 200);
    this.camera.far = Math.max(50, dist * 15);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resize() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const width = Math.max(1, parent.clientWidth);
    const height = Math.max(1, parent.clientHeight);
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /** Partial/full gold tint from grasp state: left-only / right-only / both. */
  setGraspFx({ left = false, right = false } = {}) {
    const nextLeft = Boolean(left);
    const nextRight = Boolean(right);
    if (this._graspFxLeft === nextLeft && this._graspFxRight === nextRight) return;
    this._graspFxLeft = nextLeft;
    this._graspFxRight = nextRight;
    this._reachSuccessFx = nextLeft || nextRight;
    this._applyGraspMaterials();
  }

  setReachSuccessFx(active) {
    // Backward compat: treat as full-body gold.
    const on = Boolean(active);
    this.setGraspFx({ left: on, right: on });
  }

  _refreshGraspFx() {
    this._graspFxLeft = Boolean(this._shieldAttached);
    this._graspFxRight = Boolean(this._swordAttached);
    this._reachSuccessFx = this._graspFxLeft || this._graspFxRight;
    this._applyGraspMaterials();
  }

  _meshSide(obj) {
    let cur = obj;
    while (cur) {
      const n = String(cur.name || "");
      if (n === "Sword" || n === "SwordOriented" || n === "SwordPivot" || n === "SwordHandleMarker") {
        return "right";
      }
      if (n === "Shield" || n === "ShieldOriented" || n === "ShieldPivot" || n === "ShieldHandleMarker") {
        return "left";
      }
      // Prefer URDF link names; check Right_ before Left_ so Right_Gripper_Left_* stays right.
      if (cur.isURDFLink || cur.isURDFJoint) {
        if (/^Right_/i.test(n)) return "right";
        if (/^Left_/i.test(n)) return "left";
      }
      if (/^Right_/i.test(n)) return "right";
      if (/^Left_/i.test(n)) return "left";
      cur = cur.parent;
    }
    return "center";
  }

  _forEachFxMaterial(callback) {
    const visit = (root) => {
      if (!root) return;
      root.traverse((obj) => {
        if (!obj.isMesh || !obj.material) return;
        if (obj.isURDFCollider) return;
        // Clone shared materials so left/right can be tinted independently.
        if (!obj.userData._graspFxMatUnique) {
          if (Array.isArray(obj.material)) {
            obj.material = obj.material.map((m) => (m ? m.clone() : m));
          } else if (obj.material) {
            obj.material = obj.material.clone();
          }
          obj.userData._graspFxMatUnique = true;
        }
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        for (const mat of mats) {
          if (!mat) continue;
          callback(mat, obj);
        }
      });
    };
    visit(this.robot);
    visit(this._swordOriented || this._swordMesh);
    visit(this._shieldOriented || this._shieldMesh);
  }

  _applyGraspMaterials() {
    // Bright yellow for the grasped side.
    const yellow = 0xffe14a;
    const leftOn = Boolean(this._graspFxLeft);
    const rightOn = Boolean(this._graspFxRight);
    const both = leftOn && rightOn;
    this._forEachFxMaterial((mat, obj) => {
      if (mat.userData._reachFxColor == null && mat.color) {
        mat.userData._reachFxColor = mat.color.getHex();
      }
      if (mat.emissive && mat.userData._reachFxEmissive == null) {
        mat.userData._reachFxEmissive = mat.emissive.getHex();
        mat.userData._reachFxIntensity = mat.emissiveIntensity ?? 1;
      }
      const side = this._meshSide(obj);
      // Left-only: left arm + shield. Right-only: right arm + sword. Both: entire robot + props.
      let shouldGold = false;
      if (both) shouldGold = true;
      else if (leftOn && side === "left") shouldGold = true;
      else if (rightOn && side === "right") shouldGold = true;

      if (!shouldGold) {
        if (mat.color && mat.userData._reachFxColor != null) {
          mat.color.setHex(mat.userData._reachFxColor);
        }
        if (mat.emissive && mat.userData._reachFxEmissive != null) {
          mat.emissive.setHex(mat.userData._reachFxEmissive);
          if ("emissiveIntensity" in mat) mat.emissiveIntensity = mat.userData._reachFxIntensity;
        }
      } else {
        if (mat.color) mat.color.setHex(yellow);
        if (mat.emissive) {
          mat.emissive.setHex(0x664400);
          if ("emissiveIntensity" in mat) mat.emissiveIntensity = 0.35;
        }
      }
      mat.needsUpdate = true;
    });
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    const now = performance.now();
    const deltaSec = Math.min(0.1, Math.max(0, (now - this.lastAnimationTime) / 1000));
    this.lastAnimationTime = now;
    this.updateRenderedRobot(deltaSec);
    this._updateAttachedSwordFollow();
    this.updateSwordVisualFx(deltaSec);
    this.updateShieldVisualFx(deltaSec);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.frameCounter += 1;
    if (now - this.frameWindowStart >= 1000) {
      this.fpsLabel.textContent = `${Math.round((this.frameCounter * 1000) / (now - this.frameWindowStart))} FPS`;
      this.frameCounter = 0;
      this.frameWindowStart = now;
    }
  }
}

export function bindSceneToolbar(scene) {
  document.getElementById("fitViewBtn")?.addEventListener("click", () => scene.fitView?.());
  document.getElementById("frontViewBtn")?.addEventListener("click", () => scene.frontView?.());
  document.getElementById("resetViewBtn")?.addEventListener("click", () => scene.resetView?.());
  document.getElementById("todToggleBtn")?.addEventListener("click", () => scene.toggleTimeOfDay?.());
  scene.syncTodButton?.();
}
