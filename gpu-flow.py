import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import wgpu

    import math

    return math, np, plt, wgpu


@app.cell
def _(np):
    #elevation = np.random.random((16000, 16000))
    elevation = np.load('z.npy').astype(np.float32)
    return (elevation,)


@app.cell
def _(elevation, plt):
    if elevation.shape[0] * elevation.shape[1] < 10**7:
        plt.imshow(elevation, cmap='pink')
        plt.show()
    return


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
    adapter.limits
    return


@app.cell
def _(device, elevation, math, np, wgpu):
    height, width = elevation.shape

    dimensions_buffer = device.create_buffer(
        label='dimensions',
        size=2*4,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    elevation_buffer = device.create_buffer(
        label='elevation',
        size=width * height * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    flowdir_buffer = device.create_buffer(
        label='flowdir',
        size=width * height * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    #slope_buffer = device.create_buffer(
    #    label='slope',
    #    size=width * height * 4,
    #    usage=wgpu.BufferUsage.STORAGE,
    #)

    drainage_area_buffer = device.create_buffer(
        label='drainage_area',
        size=width * height * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
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
            buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.read_only_storage),
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
        #wgpu.BindGroupLayoutEntry(
        #    binding=4,
        #    visibility=wgpu.ShaderStage.COMPUTE,
        #    buffer=wgpu.BufferBindingLayout(type=wgpu.BufferBindingType.storage),
        #),
    ])

    bind_group = device.create_bind_group(layout=bind_group_layout, entries=[
        wgpu.BindGroupEntry(binding=0, resource=dimensions_buffer),
        wgpu.BindGroupEntry(binding=1, resource=elevation_buffer),
        wgpu.BindGroupEntry(binding=2, resource=flowdir_buffer),
        wgpu.BindGroupEntry(binding=3, resource=drainage_area_buffer),
        #wgpu.BindGroupEntry(binding=4, resource=slope_buffer),
    ])

    layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])

    flowdir_shader = device.create_shader_module(code='''
    @group(0) @binding(0) var<uniform> dimensions: vec2<u32>;
    @group(0) @binding(1) var<storage, read> elevation: array<f32>;
    @group(0) @binding(2) var<storage, read_write> flowdir: array<atomic<u32>>;
    @group(0) @binding(3) var<storage, read_write> drainage_area: array<atomic<u32>>;
    // @group(0) @binding(4) var<storage, read_write> slope: array<f32>;

    var<workgroup> tile: array<array<f32, 18>, 18>;

    @compute
    @workgroup_size(16, 16, 1)
    fn main(
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) idx: u32,
    ) {
        let dims = vec2<i32>(dimensions);

        let origin = vec2<i32>(wid.xy) * 16 - vec2<i32>(1);
        for (var k = idx; k < 324u; k += 256u) {
            let t = vec2<i32>(i32(k % 18u), i32(k / 18u));
            let q = origin + t;
            var v = -999999999.0;
            if (all(q >= vec2<i32>(0)) && all(q < dims)) {
                v = elevation[q.y * dims.x + q.x];
            }
            tile[t.y][t.x] = v;
        }
        workgroupBarrier();

        if (any(gid.xy >= dimensions)) {
            return;
        }

        let center = tile[lid.y+1][lid.x+1];
        var steepest = 0.0;
        var dir = 5u;
        for (var i = 0u; i < 3; i++) {
            for (var j = 0u; j < 3; j++) {
                if (i == 1 && j == 1) {
                    continue;
                }
                let e = tile[lid.y+j][lid.x+i];
                let dx = select(1.0, sqrt(2.0), i != 1 && j != 1);
                let slope = (e - center) / dx;
                if (slope < steepest) {
                    steepest = slope;
                    dir = (i << 2) + j;
                }
            }
        }

        if (dir != 5) {
            let downstream = vec2<u32>(gid.x + (dir >> 2) - 1, gid.y + (dir & 0x3) - 1);
            if (all(downstream >= vec2<u32>(0)) && all(downstream < dimensions)) {
                atomicAdd(&flowdir[downstream.y * dimensions.x + downstream.x], 1);
            }
        }

        let i = gid.y * dimensions.x + gid.x;
        atomicOr(&flowdir[i], dir << 8);
        // slope[i] = steepest;
    }
    ''')

    flowdir_pipeline = device.create_compute_pipeline(
        layout=layout,
        compute=wgpu.ProgrammableStage(module=flowdir_shader),
    )

    flowacc_shader = device.create_shader_module(code='''
    @group(0) @binding(0) var<uniform> dimensions: vec2<u32>;
    @group(0) @binding(1) var<storage, read> elevation: array<f32>;
    @group(0) @binding(2) var<storage, read_write> flowdir: array<u32>;
    @group(0) @binding(3) var<storage, read_write> drainage_area: array<atomic<u32>>;
    // @group(0) @binding(4) var<storage, read_write> slope: array<f32>;

    @compute @workgroup_size(16, 16, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        if (any(gid.xy >= dimensions)) {
            return;
        }

        var pos = gid.xy;
        if ((flowdir[pos.y * dimensions.x + pos.x] & 0xff) != 0) {
            return;
        }

        for (var spins = 0u; spins < dimensions.x * dimensions.y; spins++) {
            let idx = pos.y * dimensions.x + pos.x;
            atomicAdd(&drainage_area[idx], 1);
            let dir = flowdir[idx] >> 8;
            pos = vec2<u32>(pos.x + (dir >> 2) - 1, pos.y + (dir & 0x3) - 1);
            // underflow catches the <0 case
            if (any(pos >= dimensions) || dir == 5) {
                return;
            }
        }
    }
    ''')

    flowacc_pipeline = device.create_compute_pipeline(
        layout=layout,
        compute=wgpu.ProgrammableStage(module=flowacc_shader),
    )

    device.queue.write_buffer(
        dimensions_buffer,
        0,
        np.array([width, height], dtype=np.uint32),
    )

    device.queue.write_buffer(
        elevation_buffer,
        0,
        np.ascontiguousarray(elevation, dtype=np.float32),
    )

    query_set = device.create_query_set(type=wgpu.QueryType.timestamp, count=2)
    timestamp_buffer = device.create_buffer(
        size=8*2,
        usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.QUERY_RESOLVE,
    )
    timestamp_map_buffer = device.create_buffer(
        size=8*2,
        usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
    )

    encoder = device.create_command_encoder()

    cpass = encoder.begin_compute_pass(timestamp_writes=wgpu.ComputePassTimestampWrites(
        query_set=query_set,
        beginning_of_pass_write_index=0,
        end_of_pass_write_index=1,
    ))
    cpass.set_bind_group(0, bind_group)

    cpass.set_pipeline(flowdir_pipeline)
    cpass.dispatch_workgroups(math.ceil(width / 16), math.ceil(height / 16), 1)

    cpass.set_pipeline(flowacc_pipeline)
    cpass.dispatch_workgroups(math.ceil(width / 16), math.ceil(height / 16), 1)

    cpass.end()

    encoder.resolve_query_set(
        query_set,
        first_query=0,
        query_count=2,
        destination=timestamp_buffer,
        destination_offset=0,
    )
    encoder.copy_buffer_to_buffer(timestamp_buffer, 0, timestamp_map_buffer, 0, 8*2)

    device.queue.submit([encoder.finish()])

    timestamp_map_buffer.map_sync(mode=wgpu.MapMode.READ)
    ts_raw = timestamp_map_buffer.read_mapped()
    timestamp_map_buffer.unmap()
    ts = np.frombuffer(ts_raw, dtype=np.uint64)
    elapsed_ns = ts[1] - ts[0]
    print(f'D8 flow accumulation on {width}x{height} grid completed in {elapsed_ns/1000000:.3f} ms')

    flowdir_raw = device.queue.read_buffer(flowdir_buffer, 0)
    flowdir = np.frombuffer(flowdir_raw, dtype=np.uint32).reshape(height, width)

    drainage_area_raw = device.queue.read_buffer(drainage_area_buffer, 0)
    drainage_area = np.frombuffer(drainage_area_raw, dtype=np.uint32).reshape(height, width)
    return drainage_area, flowdir


@app.cell
def _(elevation, flowdir, plt):
    if elevation.shape[0] * elevation.shape[1] < 10**7:
        plt.imshow(flowdir, cmap='grey')
        plt.show()
    return


@app.cell
def _(drainage_area, elevation, np, plt):
    if elevation.shape[0] * elevation.shape[1] < 10**7:
        plt.imshow(np.log10(drainage_area), cmap='magma')
        plt.show()
    return


if __name__ == "__main__":
    app.run()
