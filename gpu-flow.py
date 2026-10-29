import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import wgpu

    import math
    import time

    return math, np, plt, time, wgpu


@app.cell
def _(np):
    width = 200
    height = 100 
    nsims = 16
    np.random.seed(0)
    elevation = np.random.random((nsims, height, width))
    #elevation = np.tile(10 * (1 - np.abs(np.linspace(-1.0, 1.0, width))), (height, 1)) + np.random.random((height, width)) * 0.1
    #elevation = np.load('z.npy').astype(np.float32)
    #randbuf = np.random.random(elevation.shape)
    return elevation, height, nsims, width


@app.cell
def _(wgpu):
    adapter = wgpu.gpu.request_adapter_sync()
    device = adapter.request_device_sync(
        required_features=['timestamp-query'],
        required_limits={},
    )
    return adapter, device


@app.cell
def _(adapter):
    list(adapter.features)
    return


@app.cell
def _(adapter):
    adapter.limits
    return


@app.cell
def _(device, elevation, height, math, np, nsims, time, wgpu, width):
    dimensions_buffer = device.create_buffer(
        label='dimensions',
        size=3*4,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    #random_buffer = device.create_buffer(
    #    label='random',
    #    size=nsims * width * height * 4,
    #    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    #)

    elevation_buffer = device.create_buffer(
        label='elevation',
        size=nsims * width * height * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )

    flowdir_buffer = device.create_buffer(
        label='flowdir',
        size=nsims * width * height * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )

    drainage_area_buffer = device.create_buffer(
        label='drainage_area',
        size=nsims * width * height * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )

    elevation_out_buffer = device.create_buffer(
        label='elevation_out',
        size=nsims * width * height * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group_layout = device.create_bind_group_layout(entries=[
        wgpu.BindGroupLayoutEntry(
            binding=0,
            visibility=wgpu.ShaderStage.COMPUTE,
            buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.uniform),
        ),
        wgpu.BindGroupLayoutEntry(
            binding=1,
            visibility=wgpu.ShaderStage.COMPUTE,
            buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.storage),
        ),
        wgpu.BindGroupLayoutEntry(
            binding=2,
            visibility=wgpu.ShaderStage.COMPUTE,
            buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.storage),
        ),
        wgpu.BindGroupLayoutEntry(
            binding=3,
            visibility=wgpu.ShaderStage.COMPUTE,
            buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.storage),
        ),
        wgpu.BindGroupLayoutEntry(
            binding=4,
            visibility=wgpu.ShaderStage.COMPUTE,
            buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.storage),
        ),
        #wgpu.BindGroupLayoutEntry(
        #    binding=5,
        #    visibility=wgpu.ShaderStage.COMPUTE,
        #    buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.storage),
        #),
    ])

    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[
        wgpu.BindGroupEntry(binding=0, resource=dimensions_buffer),
        wgpu.BindGroupEntry(binding=1, resource=elevation_buffer),
        wgpu.BindGroupEntry(binding=2, resource=flowdir_buffer),
        wgpu.BindGroupEntry(binding=3, resource=drainage_area_buffer),
        wgpu.BindGroupEntry(binding=4, resource=elevation_out_buffer),
        #wgpu.BindGroupEntry(binding=5, resource=random_buffer),
    ])

    bind_group_swapped = device.create_bind_group(layout=bind_group_layout, entries=[
        wgpu.BindGroupEntry(binding=0, resource=dimensions_buffer),
        wgpu.BindGroupEntry(binding=1, resource=elevation_out_buffer),
        wgpu.BindGroupEntry(binding=2, resource=flowdir_buffer),
        wgpu.BindGroupEntry(binding=3, resource=drainage_area_buffer),
        wgpu.BindGroupEntry(binding=4, resource=elevation_buffer),
        #wgpu.BindGroupEntry(binding=5, resource=random_buffer),
    ])

    layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

    flowdir_shader = device.create_shader_module(code='''
    @group(0) @binding(0) var<uniform> dimensions: vec3u;
    @group(0) @binding(1) var<storage, read_write> elevation: array<f32>;
    // packed: in-degree in LS byte, flowdir in second byte
    @group(0) @binding(2) var<storage, read_write> flowdir: array<atomic<u32>>;
    // packed: in-degree in top 4 bits
    @group(0) @binding(3) var<storage, read_write> drainage_area: array<atomic<u32>>;
    @group(0) @binding(4) var<storage, read_write> elevation_out: array<f32>;

    var<workgroup> tile: array<array<f32, 18>, 18>;

    @compute
    @workgroup_size(1, 16, 16)
    fn main(
        @builtin(workgroup_id) wid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_index) idx: u32,
    ) {
        let dims = vec3<i32>(dimensions);
        let pos = gid.yz;
        let offset = gid.x * dimensions.y * dimensions.z;

        let origin = vec2<i32>(wid.yz) * 16 - vec2<i32>(1);
        for (var k = idx; k < 324u; k += 256u) {
            let t = vec2<i32>(i32(k % 18u), i32(k / 18u));
            let q = origin + t;
            var v = 0.0;
            // boundary conditions
            if (q.y < 0 || q.y >= dims.z) {
                v = 999999999.0;
            }
            if (all(q >= vec2<i32>(0)) && all(q < dims.yz)) {
                v = elevation[i32(offset) + q.y * dims.y + q.x];
            }
            tile[t.y][t.x] = v;
        }
        workgroupBarrier();

        if (any(gid >= dimensions)) {
            return;
        }

        let center = tile[lid.z+1][lid.y+1];
        var steepest = 0.0;
        var dir = 5u;
        for (var i = 0u; i < 3; i++) {
            for (var j = 0u; j < 3; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                let e = tile[lid.z+j][lid.y+i];
                let dx = select(1.0, sqrt(2.0), i != 1 && j != 1);
                let slope = (e - center) / dx;
                if (slope < steepest) {
                    steepest = slope;
                    dir = (i << 2) + j;
                }
            }
        }

        if (dir != 5) {
            let downstream = vec2<u32>(pos.x + (dir >> 2) - 1, pos.y + (dir & 0x3) - 1);
            if (all(downstream >= vec2<u32>(0)) && all(downstream < dimensions.yz)) {
                atomicAdd(&flowdir[offset + downstream.y * dimensions.y + downstream.x], 1);
                atomicAdd(&drainage_area[offset + downstream.y * dimensions.y + downstream.x], 1 << 28);
            }
        }

        let i = offset + pos.y * dimensions.y + pos.x;
        atomicOr(&flowdir[i], dir << 8);
        // slope[i] = steepest;
        atomicAdd(&drainage_area[i], 1);
    }
    ''')

    flowdir_pipeline = device.create_compute_pipeline(
        layout=layout,
        compute=wgpu.ProgrammableStage(module=flowdir_shader),
    )

    flowacc_shader = device.create_shader_module(code='''
    @group(0) @binding(0) var<uniform> dimensions: vec3u;
    @group(0) @binding(1) var<storage, read_write> elevation: array<f32>;
    // packed: in-degree in LS byte, flowdir in second byte
    @group(0) @binding(2) var<storage, read_write> flowdir: array<u32>;
    // packed: in-degree in top 4 bits
    @group(0) @binding(3) var<storage, read_write> drainage_area: array<atomic<u32>>;
    @group(0) @binding(4) var<storage, read_write> elevation_out: array<f32>;

    const SHIFT = 28u;
    const MASK = 0xfu << SHIFT;

    @compute @workgroup_size(1, 16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
        if (any(gid.xyz >= dimensions)) {
            return;
        }

        let offset = gid.x * dimensions.y * dimensions.z;

        var pos = gid.yz;
        // exit if in-degree > 0
        if ((flowdir[offset + pos.y * dimensions.y + pos.x] & 0xff) > 0) {
            return;
        }

        var area = 1u;
        for (var spins = 0u; spins < dimensions.y * dimensions.z; spins++) {
            let idx = offset + pos.y * dimensions.y + pos.x;
            if (spins > 0) {
                // the last thread to visit a cell propagates the upstream area, all others exit.
                let prev = atomicSub(&drainage_area[idx], (1u << SHIFT) - area);
                let indegree = (prev & MASK) >> SHIFT;
                area += prev & ~MASK;
                if (indegree > 1) {
                    return;
                }
            }

            let dir = flowdir[idx] >> 8;
            pos = vec2<u32>(pos.x + (dir >> 2) - 1, pos.y + (dir & 0x3) - 1);
            // underflow catches the <0 case
            if (any(pos >= dimensions.yz) || dir == 5) {
                return;
            }
        }
    }
    ''')

    flowacc_pipeline = device.create_compute_pipeline(
        layout=layout,
        compute=wgpu.ProgrammableStage(module=flowacc_shader),
    )

    spim_shader = device.create_shader_module(code='''
    @group(0) @binding(0) var<uniform> dimensions: vec3u;
    @group(0) @binding(1) var<storage, read_write> elevation: array<f32>;
    // packed: in-degree in LS byte, flowdir in second byte
    @group(0) @binding(2) var<storage, read_write> flowdir: array<u32>;
    @group(0) @binding(3) var<storage, read_write> drainage_area: array<u32>;
    @group(0) @binding(4) var<storage, read_write> elevation_out: array<f32>;

    @compute
    @workgroup_size(1, 16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3u) {
        if (any(gid >= dimensions)) {
            return;
        }
        let pos = gid.yz;
        let offset = gid.x * dimensions.y * dimensions.z;
        let idx = offset + pos.y * dimensions.y + pos.x;

        let k = 0.01;
        let m = 0.5;
        let n = 1.0;
        let u = 0.01 ;//+ random[idx] * 0.005;
        let dt = 1.0;

        let z = elevation[idx];

        let dir = flowdir[idx] >> 8;
        let downstream = vec2<u32>(pos.x + (dir >> 2) - 1, pos.y + (dir & 0x3) - 1);
        if (any(downstream >= dimensions.yz)) {
            elevation_out[idx] = elevation[idx];
            return;
        }
        let downstream_idx = offset + downstream.y * dimensions.y + downstream.x;
        let downstream_z = elevation[downstream_idx];
        let a = f32(drainage_area[idx]);
        let dx = select(1.0, sqrt(2.0), (dir >> 2) != 1 && (dir & 0x3) != 1) * 1.0;
        let dzdx = -(downstream_z - z) / dx;

        let dzdt = u - k * pow(a,m) * pow(dzdx,n);
        let dz = dzdt * dt;
        elevation_out[idx] = max(z + dz, 0.0);
        if (pos.x == 0 || pos.x == dimensions.y-1) {
            elevation_out[idx] = 0.0;
        }
    }
    ''')

    spim_pipeline = device.create_compute_pipeline(
        layout=layout,
        compute=wgpu.ProgrammableStage(module=spim_shader),
    )

    device.queue.write_buffer(
        dimensions_buffer,
        0,
        np.array([nsims, width, height], dtype=np.uint32),
    )

    device.queue.write_buffer(
        elevation_buffer,
        0,
        np.ascontiguousarray(elevation, dtype=np.float32),
    )

    #device.queue.write_buffer(
    #    random_buffer,
    #    0,
    #    np.ascontiguousarray(randbuf, dtype=np.float32),
    #)

    n = 4000
    group_size = 200

    query_set = device.create_query_set(type=wgpu.QueryType.timestamp, count=2)
    timestamp_buffer = device.create_buffer(
        size=8*2,
        usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.QUERY_RESOLVE,
    )
    timestamp_map_buffer = device.create_buffer(
        size=8*2*group_size,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )

    total_ns = 0
    wall_ns = 0
    start = time.perf_counter_ns()
    swap_bind_group = False
    for i in range(max(1, n // group_size)):
        encoder = device.create_command_encoder()
        for j in range(min(n - i*group_size, group_size)):
            encoder.clear_buffer(flowdir_buffer)
            encoder.clear_buffer(drainage_area_buffer)

            cpass = encoder.begin_compute_pass(timestamp_writes=wgpu.ComputePassTimestampWrites(
                query_set=query_set,
                beginning_of_pass_write_index=0,
                end_of_pass_write_index=1,
            ))
            if swap_bind_group:
                cpass.set_bind_group(0, bind_group_swapped)
            else:
                cpass.set_bind_group(0, bind_group)
            swap_bind_group = not swap_bind_group

            wg_dims = nsims, math.ceil(width / 16), math.ceil(height / 16)

            cpass.set_pipeline(flowdir_pipeline)
            cpass.dispatch_workgroups(*wg_dims)

            cpass.set_pipeline(flowacc_pipeline)
            cpass.dispatch_workgroups(*wg_dims)

            cpass.set_pipeline(spim_pipeline)
            cpass.dispatch_workgroups(*wg_dims)

            cpass.end()

            encoder.resolve_query_set(
                query_set,
                first_query=0,
                query_count=2,
                destination=timestamp_buffer,
                destination_offset=0,
            )
            encoder.copy_buffer_to_buffer(timestamp_buffer, 0, timestamp_map_buffer, j*8*2, 8*2)

        t0 = time.perf_counter_ns()
        device.queue.submit([encoder.finish()])

        timestamp_map_buffer.map_sync(mode=wgpu.MapMode.READ)
        wall_ns += time.perf_counter_ns() - t0
        ts_raw = timestamp_map_buffer.read_mapped()
        timestamp_map_buffer.unmap()
        ts = np.frombuffer(ts_raw, dtype=np.uint64).reshape(group_size, 2)
        elapsed_ns = np.sum(ts[:, 1] - ts[:, 0])
        total_ns += elapsed_ns

    print(f'Ran {n} simulation steps for {nsims} simulations on a {width}x{height} grid in {wall_ns/1000000:.3f} ms')
    x = time.perf_counter_ns() - start
    #print(f'{x/1000000:.3f} ms')

    flowdir_raw = device.queue.read_buffer(flowdir_buffer, 0)
    flowdir = np.frombuffer(flowdir_raw, dtype=np.uint32).reshape(nsims, height, width)

    drainage_area_raw = device.queue.read_buffer(drainage_area_buffer, 0)
    drainage_area = np.frombuffer(drainage_area_raw, dtype=np.uint32).reshape(nsims, height, width)

    elevation_raw = device.queue.read_buffer(elevation_out_buffer if swap_bind_group else elevation_buffer, 0)
    z = np.frombuffer(elevation_raw, dtype=np.float32).reshape(nsims, height, width)
    return drainage_area, flowdir, z


@app.cell
def _(math, np):
    def grid(a, gap=2):
        # arrange (nsims, height, width) into a near-square 2d grid, NaN-padded
        n, h, w = a.shape
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        out = np.full((rows * (h + gap) - gap, cols * (w + gap) - gap), np.nan)
        for k in range(n):
            r, c = divmod(k, cols)
            out[r*(h+gap):r*(h+gap)+h, c*(w+gap):c*(w+gap)+w] = a[k]
        return out

    return (grid,)


@app.cell
def _(elevation, grid, plt, z):
    if elevation.size < 10**7:
        plt.imshow(grid(z), cmap='magma')
        plt.gcf().set_size_inches(10, 10)
        plt.show()
    return


@app.cell
def _(np, z):
    np.count_nonzero(z == 0.0)
    return


@app.cell
def _(z):
    z.mean()
    return


@app.cell
def _(elevation, flowdir, grid, plt):
    if elevation.size < 10**7:
        plt.imshow(grid(flowdir >> 8), cmap='grey')
        plt.gcf().set_size_inches(10, 10)
        plt.show()
    return


@app.cell
def _(drainage_area, elevation, grid, np, plt):
    if elevation.size < 10**7:
        plt.imshow(grid(np.log10(drainage_area)), cmap='magma')
        plt.gcf().set_size_inches(10, 10)
        plt.show()
    return


@app.cell
def _(drainage_area, np):
    np.max(drainage_area)
    return


if __name__ == "__main__":
    app.run()
