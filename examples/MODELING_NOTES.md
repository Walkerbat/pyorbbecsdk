# Orbbec 相机给工件建模 —— 使用与 MuJoCo 集成说明

整条链路分三步，全部在本目录（`pyorbbecsdk/examples/`）内：

```
capture_scan.py   ->  多视角扫描，逐帧存 pointcloud.ply
reconstruct.py    ->  ICP 配准 + Poisson 重建 + 导出 mesh.stl + dims.json
scene.xml         ->  把 STL 挂进 MuJoCo，填物理参数
```

环境：`conda activate bin_grasp`（Python 3.9，含 open3d 0.19 + pyorbbecsdk + opencv）。

---

## 1. 扫描采集

```bash
python capture_scan.py --out scan_my_part
```

- 相机固定、工件放转台自转 → 转台模式（默认）。
- 相机绕静止工件转 → 按 `r` 切成手持模式。
- 预览窗口里按 **空格** 抓一帧，按 `s` 结束并打印清单。

每帧落盘：`pointcloud.ply`（带色点云）+ `color.png` + `depth.png` + `intrinsics.npy`。

**扫描建议**：8~24 帧，相邻帧有 30% 以上重叠；工件表面若是纯色/反光金属，
优先转台模式（绕 z 旋转的强先验让配准很稳），并加 `--with-color` 用彩色 ICP。

## 2. 配准重建导出

```bash
python reconstruct.py --scan scan_my_part --turntable --voxel 0.0015 [--with-color] [--crop-bottom 0.005]
```

产出：
- `fused.ply`  融合点云
- `mesh.ply` / `mesh.stl`  三角网格（STL 直接进 MuJoCo）
- `dims.json`  AABB 长宽高、中心、体积 —— 填 scene.xml 用

`--crop-bottom`：转台工件的底面被转台挡住没扫到，Poisson 会封一个近似的平底；
若需精确底，用这个参数按 AABB 底部裁剪后再自行封底。

## 3. 挂进 MuJoCo

复制 STL 到仿真网格目录：

```bash
cp scan_my_part/mesh.stl ../Bin_grasp/simulation/meshes/my_part.stl
```

改 `Bin_grasp/simulation/scene.xml` 的 `<asset>`，加一行 mesh：

```xml
<mesh name="my_part" file="meshes/my_part.stl"/>
```

把某个 `workpiece_N` 的 box geom 换成 mesh（mesh 的 mass 写在 geom 上，密度
由 MuJoCo 从体积自动算）：

```xml
<body name="workpiece_0" pos="..." quat="...">
  <freejoint name="workpiece_0_free"/>
  <geom type="mesh" mesh="my_part" mass="0.15" material="part_mat"/>
  <!-- grasp_arrow_* 标记保持不变 -->
</body>
```

### 需要同步改的三个物理假设（关键！）

重建件一旦不是「顶面水平的扁平矩形」，`run_mujoco.py` 和
`suctionnet_bridge.py` 里的这些假设会同时失效，必须一起改：

1. **半高** `PART_HALF_Z = 0.015`（`run_mujoco.py:29`）
   从 `dims.json` 的 `extent_xyz[2] / 2` 取真实半高。若工件顶面不是平面，
   抓取/放置的 z 偏移要改成「顶面到底面的真实距离」。

2. **吸盘-顶面接触面积** `_disk_rect_area`（`run_mujoco.py:319`）
   现在用「圆盘 × 矩形」裁剪算接触面积。异形顶面（圆/曲面/有孔）必须换成
   圆盘与真实顶面形状的交集，否则吸力 `suction_hold_force` 算错。

3. **顶面法向朝上的假设**（`suctionnet_bridge.py` 的法向估计 + 抓取 approach）
   斜面/曲面件的抓取方向不再是 +z，法向估计和 `approach = grasp_point + normal*0.22`
   要按真实表面法向走（这段本身已经用预测法向了，但「顶面朝上」的兜底
   `[0,0,1]` 只对扁平件成立）。

**结论**：如果是「扁平件、尺寸变化」，改 1 处 XML + `PART_HALF_Z` 即可；
如果是异形件，上面的 2、3 是真正的工作量所在——扫描重建只解决「几何」，
「可抓取的物理模型」还需要把接触/法向模型泛化到新形状。

---

## 附：离线自检（不插相机）

```bash
python offline_demo.py
python reconstruct.py --scan offline_demo_scan --turntable --voxel 0.002
```

会生成一个合成盒子的 12 帧扫描并完整走一遍重建，用于确认 open3d 环境正常。
