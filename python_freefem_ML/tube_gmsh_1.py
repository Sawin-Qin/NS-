import gmsh

gmsh.initialize()
gmsh.model.add("rect_5x1_fluid_solid")

# ==================== 参数（你只调这里） ====================
L_out = 5.0        # 外矩形长 (m)  —— 固定 5m
W_out = 1.0        # 外矩形宽 (m)  —— 固定 1m
L_in  = 1.0        # 内矩形长 (m)  —— 固定 1m
W_in  = 0.3        # ★ 内矩形宽 (m) —— 变量，待定，改这一行

mesh_fluid = 0.06   # 流体域网格尺寸（粗一点，控总量）
mesh_solid = 0.04   # 固体域网格尺寸
# =================================================================

# ---------- 1. 外矩形（整个域的外边界） ----------
p1 = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, mesh_fluid)
p2 = gmsh.model.occ.addPoint(L_out, 0.0, 0.0, mesh_fluid)
p3 = gmsh.model.occ.addPoint(L_out, W_out, 0.0, mesh_fluid)
p4 = gmsh.model.occ.addPoint(0.0, W_out, 0.0, mesh_fluid)

l1 = gmsh.model.occ.addLine(p1, p2)   # 下边
l2 = gmsh.model.occ.addLine(p2, p3)   # 右边
l3 = gmsh.model.occ.addLine(p3, p4)   # 上边
l4 = gmsh.model.occ.addLine(p4, p1)   # 左边
outer_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])

# ---------- 2. 内矩形（居中 + 关于 y=0.5 对称） ----------
cx = L_out / 2.0            # x 中心 = 2.5
cy = W_out / 2.0            # y 中心 = 0.5
x0 = cx - L_in / 2.0        # = 2.0
y0 = cy - W_in / 2.0

r1 = gmsh.model.occ.addPoint(x0, y0, 0.0, mesh_solid)
r2 = gmsh.model.occ.addPoint(x0 + L_in, y0, 0.0, mesh_solid)
r3 = gmsh.model.occ.addPoint(x0 + L_in, y0 + W_in, 0.0, mesh_solid)
r4 = gmsh.model.occ.addPoint(x0, y0 + W_in, 0.0, mesh_solid)

rl1 = gmsh.model.occ.addLine(r1, r2)   # 内矩形下边
rl2 = gmsh.model.occ.addLine(r2, r3)   # 右边
rl3 = gmsh.model.occ.addLine(r3, r4)   # 上边
rl4 = gmsh.model.occ.addLine(r4, r1)   # 左边
inner_loop = gmsh.model.occ.addCurveLoop([rl1, rl2, rl3, rl4])

# ---------- 3. 流体域 = 外矩形 挖掉 内矩形（带孔面） ----------
fluid_surf = gmsh.model.occ.addPlaneSurface([outer_loop, inner_loop])

# ---------- 4. 固体域 = 内矩形本身（独立面） ----------
solid_surf = gmsh.model.occ.addPlaneSurface([inner_loop])

gmsh.model.occ.synchronize()

# ---------- 5. 物理组（FNO / 求解器直接认） ----------
# 流体域
gmsh.model.addPhysicalGroup(2, [fluid_surf], tag=1)
gmsh.model.setPhysicalName(2, 1, "fluid")

# 固体域
gmsh.model.addPhysicalGroup(2, [solid_surf], tag=2)
gmsh.model.setPhysicalName(2, 2, "solid")

# 入口（外矩形左边）
gmsh.model.addPhysicalGroup(1, [l4], tag=3)
gmsh.model.setPhysicalName(1, 3, "inlet")

# 出口（外矩形右边）
gmsh.model.addPhysicalGroup(1, [l2], tag=4)
gmsh.model.setPhysicalName(1, 4, "outlet")

# 外边界上下壁面
gmsh.model.addPhysicalGroup(1, [l1, l3], tag=5)
gmsh.model.setPhysicalName(1, 5, "outer_wall")

# 内矩形壁面（流体-固体交界面，FSI 关键）
gmsh.model.addPhysicalGroup(1, [rl1, rl2, rl3, rl4], tag=6)
gmsh.model.setPhysicalName(1, 6, "interface")

# ---------- 6. 网格 ----------
gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)
gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_fluid)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.model.mesh.generate(2)

# ---------- 7. 统计 ----------
nodes = gmsh.model.mesh.getNodes()
n_nodes = len(nodes[0])
tri = gmsh.model.mesh.getElementsByType(2)
n_tri = len(tri[0]) // 3 if len(tri[0]) > 0 else 0

area_fluid = 5.0 * 1.0 - 1.0 * W_in
area_solid = 1.0 * W_in

print("=" * 46)
print(f"外矩形      : {L_out} x {W_out} = {L_out*W_out} m²")
print(f"内矩形(固体): {L_in} x {W_in} = {area_solid} m²  (变量 W_in)")
print(f"流体域面积  : {area_fluid} m²")
print(f"节点数      : {n_nodes}")
print(f"三角形数    : {n_tri}")
print("=" * 46)

gmsh.write("rect_5x1_fsi.msh")
gmsh.fltk.run()
gmsh.finalize()