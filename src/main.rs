//! Illustrates rendering using D3D11. Supports Windows with D3D11 capable hardware.
//!
//! Renders a spinning RGB triangle 1 meter in front of the user.
//!
//! This example uses minimal abstraction for clarity. Real-world code should encapsulate and
//! largely decouple its D3D11 and OpenXR components and handle errors gracefully.

use std::mem;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use openxr as xr;
use windows::{
    Win32::Foundation::*, Win32::Graphics::Direct3D::Fxc::*, Win32::Graphics::Direct3D::*,
    Win32::Graphics::Direct3D11::*, Win32::Graphics::Dxgi::Common::*, core::*,
};

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct TransformBuffer {
    view_proj: [[f32; 16]; 2], // Two 4x4 matrices for stereo
    model: [f32; 16],          // 4x4 model matrix
}

// Helper function to create model matrix from position and scale
fn model_matrix_from_pose(pos: &xr::Vector3f, quat: &xr::Quaternionf, scale: f32) -> [f32; 16] {
    let xx = quat.x * quat.x;
    let yy = quat.y * quat.y;
    let zz = quat.z * quat.z;
    let xy = quat.x * quat.y;
    let xz = quat.x * quat.z;
    let yz = quat.y * quat.z;
    let wx = quat.w * quat.x;
    let wy = quat.w * quat.y;
    let wz = quat.w * quat.z;

    [
        scale * (1.0 - 2.0 * (yy + zz)),
        scale * (2.0 * (xy + wz)),
        scale * (2.0 * (xz - wy)),
        0.0,
        scale * (2.0 * (xy - wz)),
        scale * (1.0 - 2.0 * (xx + zz)),
        scale * (2.0 * (yz + wx)),
        0.0,
        scale * (2.0 * (xz + wy)),
        scale * (2.0 * (yz - wx)),
        scale * (1.0 - 2.0 * (xx + yy)),
        0.0,
        pos.x,
        pos.y,
        pos.z,
        1.0,
    ]
}

pub fn main() {
    // Handle interrupts gracefully
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::Relaxed);
    })
    .expect("setting Ctrl-C handler");

    #[cfg(all(windows, feature = "static"))]
    #[link(name = "advapi32")]
    unsafe extern "C" {}

    #[cfg(feature = "static")]
    let entry = xr::Entry::linked();
    #[cfg(not(feature = "static"))]
    let entry = unsafe {
        xr::Entry::load()
            .expect("couldn't find the OpenXR loader; try enabling the \"static\" feature")
    };

    let available_extensions = entry.enumerate_extensions().unwrap();
    assert!(available_extensions.khr_d3d11_enable);

    let mut enabled_extensions = xr::ExtensionSet::default();
    enabled_extensions.khr_d3d11_enable = true;

    let xr_instance = entry
        .create_instance(
            &xr::ApplicationInfo {
                application_name: "d3d11-openxr-example",
                application_version: 0,
                engine_name: "d3d11-openxr-example",
                engine_version: 0,
                api_version: xr::Version::new(1, 0, 0),
            },
            &enabled_extensions,
            &[],
        )
        .unwrap();

    let instance_props = xr_instance.properties().unwrap();
    println!(
        "loaded OpenXR runtime: {} {}",
        instance_props.runtime_name, instance_props.runtime_version
    );

    let system = xr_instance
        .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .unwrap();

    let environment_blend_mode = xr_instance
        .enumerate_environment_blend_modes(system, VIEW_TYPE)
        .unwrap()[0];

    let requirements = xr_instance
        .graphics_requirements::<xr::D3D11>(system)
        .unwrap();

    println!(
        "D3D11 min feature level: {:?}",
        requirements.min_feature_level
    );

    unsafe {
        let feature_levels = [D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0];

        let mut device: Option<ID3D11Device> = None;
        let mut device_context: Option<ID3D11DeviceContext> = None;
        let mut feature_level = D3D_FEATURE_LEVEL_11_0;

        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&feature_levels),
            D3D11_SDK_VERSION,
            Some(&mut device),
            Some(&mut feature_level),
            Some(&mut device_context),
        )
        .expect("Failed to create D3D11 device");

        let device = device.unwrap();
        let device_context = device_context.unwrap();

        println!(
            "Created D3D11 device with feature level: {:?}",
            feature_level
        );

        let (session, mut frame_wait, mut frame_stream) = xr_instance
            .create_session::<xr::D3D11>(
                system,
                &xr::d3d::SessionCreateInfoD3D11 {
                    device: device.as_raw() as *mut _,
                },
            )
            .unwrap();

        let action_set = xr_instance
            .create_action_set("input", "input pose information", 0)
            .unwrap();

        let right_action = action_set
            .create_action::<xr::Posef>("right_hand", "Right Hand Controller", &[])
            .unwrap();
        let left_action = action_set
            .create_action::<xr::Posef>("left_hand", "Left Hand Controller", &[])
            .unwrap();

        xr_instance
            .suggest_interaction_profile_bindings(
                xr_instance
                    .string_to_path("/interaction_profiles/khr/simple_controller")
                    .unwrap(),
                &[
                    xr::Binding::new(
                        &right_action,
                        xr_instance
                            .string_to_path("/user/hand/right/input/grip/pose")
                            .unwrap(),
                    ),
                    xr::Binding::new(
                        &left_action,
                        xr_instance
                            .string_to_path("/user/hand/left/input/grip/pose")
                            .unwrap(),
                    ),
                ],
            )
            .unwrap();

        session.attach_action_sets(&[&action_set]).unwrap();

        let right_space = right_action
            .create_space(&session, xr::Path::NULL, xr::Posef::IDENTITY)
            .unwrap();
        let left_space = left_action
            .create_space(&session, xr::Path::NULL, xr::Posef::IDENTITY)
            .unwrap();

        let stage = session
            .create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)
            .unwrap();

        // Compile shaders
        let shader_code = include_str!("d3d11_triangle.hlsl");
        let vs_blob = compile_shader(shader_code, "VSMain", "vs_5_0");
        let ps_blob = compile_shader(shader_code, "PSMain", "ps_5_0");

        let vs_bytecode = std::slice::from_raw_parts(
            vs_blob.GetBufferPointer() as *const u8,
            vs_blob.GetBufferSize(),
        );
        let mut vertex_shader: Option<ID3D11VertexShader> = None;
        device
            .CreateVertexShader(vs_bytecode, None, Some(&mut vertex_shader))
            .expect("Failed to create vertex shader");
        let vertex_shader = vertex_shader.unwrap();

        let ps_bytecode = std::slice::from_raw_parts(
            ps_blob.GetBufferPointer() as *const u8,
            ps_blob.GetBufferSize(),
        );
        let mut pixel_shader: Option<ID3D11PixelShader> = None;
        device
            .CreatePixelShader(ps_bytecode, None, Some(&mut pixel_shader))
            .expect("Failed to create pixel shader");
        let pixel_shader = pixel_shader.unwrap();

        // Create input layout for vertex data
        let input_layout_desc = [
            D3D11_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(c"POSITION".as_ptr() as _),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 0,
                InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
            D3D11_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(c"COLOR".as_ptr() as _),
                SemanticIndex: 0,
                Format: DXGI_FORMAT_R32G32B32_FLOAT,
                InputSlot: 0,
                AlignedByteOffset: 12,
                InputSlotClass: D3D11_INPUT_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            },
        ];

        let mut input_layout: Option<ID3D11InputLayout> = None;
        device
            .CreateInputLayout(&input_layout_desc, vs_bytecode, Some(&mut input_layout))
            .expect("Failed to create input layout");
        let input_layout = input_layout.unwrap();

        // Create vertex buffer with RGB triangle
        let vertices = [
            Vertex {
                position: [0.0, 0.3, 0.0],
                color: [1.0, 0.0, 0.0], // Red
            },
            Vertex {
                position: [-0.3, -0.3, 0.0],
                color: [0.0, 1.0, 0.0], // Green
            },
            Vertex {
                position: [0.3, -0.3, 0.0],
                color: [0.0, 0.0, 1.0], // Blue
            },
        ];

        let vertex_buffer_desc = D3D11_BUFFER_DESC {
            ByteWidth: mem::size_of_val(&vertices) as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_VERTEX_BUFFER.0 as u32,
            CPUAccessFlags: Default::default(),
            MiscFlags: Default::default(),
            StructureByteStride: 0,
        };

        let vertex_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: vertices.as_ptr() as *const _,
            SysMemPitch: 0,
            SysMemSlicePitch: 0,
        };

        let mut vertex_buffer: Option<ID3D11Buffer> = None;
        device
            .CreateBuffer(
                &vertex_buffer_desc,
                Some(&vertex_data),
                Some(&mut vertex_buffer),
            )
            .expect("Failed to create vertex buffer");
        let vertex_buffer = vertex_buffer.unwrap();

        // Create constant buffer for transforms
        let constant_buffer_desc = D3D11_BUFFER_DESC {
            ByteWidth: mem::size_of::<TransformBuffer>() as u32,
            Usage: D3D11_USAGE_DYNAMIC,
            BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
            CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
            MiscFlags: Default::default(),
            StructureByteStride: 0,
        };

        let mut constant_buffer: Option<ID3D11Buffer> = None;
        device
            .CreateBuffer(&constant_buffer_desc, None, Some(&mut constant_buffer))
            .expect("Failed to create constant buffer");
        let constant_buffer = constant_buffer.unwrap();

        // Create pipeline states
        let blend_desc = D3D11_BLEND_DESC {
            AlphaToCoverageEnable: FALSE,
            IndependentBlendEnable: FALSE,
            RenderTarget: [
                D3D11_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: FALSE,
                    SrcBlend: D3D11_BLEND_ONE,
                    DestBlend: D3D11_BLEND_ZERO,
                    BlendOp: D3D11_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D11_BLEND_ONE,
                    DestBlendAlpha: D3D11_BLEND_ZERO,
                    BlendOpAlpha: D3D11_BLEND_OP_ADD,
                    RenderTargetWriteMask: D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8,
                },
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
            ],
        };
        let mut blend_state: Option<ID3D11BlendState> = None;
        device
            .CreateBlendState(&blend_desc, Some(&mut blend_state))
            .expect("Failed to create blend state");
        let blend_state = blend_state.unwrap();

        let rasterizer_desc = D3D11_RASTERIZER_DESC {
            FillMode: D3D11_FILL_SOLID,
            CullMode: D3D11_CULL_NONE,
            FrontCounterClockwise: FALSE,
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
            DepthClipEnable: TRUE,
            ScissorEnable: FALSE,
            MultisampleEnable: FALSE,
            AntialiasedLineEnable: FALSE,
        };
        let mut rasterizer_state: Option<ID3D11RasterizerState> = None;
        device
            .CreateRasterizerState(&rasterizer_desc, Some(&mut rasterizer_state))
            .expect("Failed to create rasterizer state");
        let rasterizer_state = rasterizer_state.unwrap();

        let depth_stencil_desc = D3D11_DEPTH_STENCIL_DESC {
            DepthEnable: TRUE,
            DepthWriteMask: D3D11_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D11_COMPARISON_LESS,
            StencilEnable: FALSE,
            StencilReadMask: 0,
            StencilWriteMask: 0,
            FrontFace: Default::default(),
            BackFace: Default::default(),
        };
        let mut depth_stencil_state: Option<ID3D11DepthStencilState> = None;
        device
            .CreateDepthStencilState(&depth_stencil_desc, Some(&mut depth_stencil_state))
            .expect("Failed to create depth stencil state");
        let depth_stencil_state = depth_stencil_state.unwrap();

        // Main loop
        let mut swapchain: Option<Swapchain> = None;
        let mut event_storage = xr::EventDataBuffer::new();
        let mut session_running = false;
        let start_time = std::time::Instant::now();

        'main_loop: loop {
            if !running.load(Ordering::Relaxed) {
                println!("requesting exit");
                match session.request_exit() {
                    Ok(()) => {}
                    Err(xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => break,
                    Err(e) => panic!("{}", e),
                }
            }

            while let Some(event) = xr_instance.poll_event(&mut event_storage).unwrap() {
                use xr::Event::*;
                match event {
                    SessionStateChanged(e) => {
                        println!("entered state {:?}", e.state());
                        match e.state() {
                            xr::SessionState::READY => {
                                session.begin(VIEW_TYPE).unwrap();
                                session_running = true;
                            }
                            xr::SessionState::STOPPING => {
                                session.end().unwrap();
                                session_running = false;
                            }
                            xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                                break 'main_loop;
                            }
                            _ => {}
                        }
                    }
                    InstanceLossPending(_) => break 'main_loop,
                    EventsLost(e) => println!("lost {} events", e.lost_event_count()),
                    _ => {}
                }
            }

            if !session_running {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }

            let xr_frame_state = frame_wait.wait().unwrap();
            frame_stream.begin().unwrap();

            if !xr_frame_state.should_render {
                frame_stream
                    .end(
                        xr_frame_state.predicted_display_time,
                        environment_blend_mode,
                        &[],
                    )
                    .unwrap();
                continue;
            }

            let swapchain = swapchain.get_or_insert_with(|| {
                let views = xr_instance
                    .enumerate_view_configuration_views(system, VIEW_TYPE)
                    .unwrap();
                assert_eq!(views.len(), VIEW_COUNT as usize);

                let resolution = (
                    views[0].recommended_image_rect_width,
                    views[0].recommended_image_rect_height,
                );

                let handle = session
                    .create_swapchain(&xr::SwapchainCreateInfo {
                        create_flags: xr::SwapchainCreateFlags::EMPTY,
                        usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                            | xr::SwapchainUsageFlags::SAMPLED,
                        format: COLOR_FORMAT,
                        sample_count: 1,
                        width: resolution.0,
                        height: resolution.1,
                        face_count: 1,
                        array_size: VIEW_COUNT,
                        mip_count: 1,
                    })
                    .unwrap();

                let images = handle.enumerate_images().unwrap();
                let render_target_views = images
                    .iter()
                    .map(|&texture_ptr| {
                        let texture: ID3D11Texture2D = ID3D11Texture2D::from_raw(texture_ptr as _);

                        let rtv_desc = D3D11_RENDER_TARGET_VIEW_DESC {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                            ViewDimension: D3D11_RTV_DIMENSION_TEXTURE2DARRAY,
                            Anonymous: D3D11_RENDER_TARGET_VIEW_DESC_0 {
                                Texture2DArray: D3D11_TEX2D_ARRAY_RTV {
                                    MipSlice: 0,
                                    FirstArraySlice: 0,
                                    ArraySize: VIEW_COUNT,
                                },
                            },
                        };

                        let mut rtv: Option<ID3D11RenderTargetView> = None;
                        device
                            .CreateRenderTargetView(&texture, Some(&rtv_desc), Some(&mut rtv))
                            .expect("Failed to create render target view");

                        mem::forget(texture);
                        rtv.unwrap()
                    })
                    .collect::<Vec<_>>();

                Swapchain {
                    handle,
                    resolution,
                    render_target_views,
                }
            });

            let image_index = swapchain.handle.acquire_image().unwrap();
            swapchain.handle.wait_image(xr::Duration::INFINITE).unwrap();

            // Calculate rotation based on elapsed time
            let elapsed = start_time.elapsed().as_secs_f32();
            let rotation_angle = elapsed * 0.5; // Rotate at 0.5 radians per second

            let (_, views) = session
                .locate_views(VIEW_TYPE, xr_frame_state.predicted_display_time, &stage)
                .unwrap();

            // Update constant buffer with view-projection matrices and model matrix
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            device_context
                .Map(
                    &constant_buffer,
                    0,
                    D3D11_MAP_WRITE_DISCARD,
                    0,
                    Some(&mut mapped),
                )
                .expect("Failed to map constant buffer");

            let transform_data = mapped.pData as *mut TransformBuffer;
            (*transform_data).view_proj[0] = compute_view_proj_matrix(&views[0]);
            (*transform_data).view_proj[1] = compute_view_proj_matrix(&views[1]);
            (*transform_data).model = compute_model_matrix(rotation_angle);

            device_context.Unmap(&constant_buffer, 0);

            // Render
            let rtv = &swapchain.render_target_views[image_index as usize];
            let clear_color = [0.0f32, 0.0, 0.0, 1.0];
            device_context.ClearRenderTargetView(rtv, &clear_color);

            device_context.OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);

            let viewport = D3D11_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: swapchain.resolution.0 as f32,
                Height: swapchain.resolution.1 as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            device_context.RSSetViewports(Some(&[viewport]));

            device_context.IASetInputLayout(&input_layout);
            device_context.VSSetShader(&vertex_shader, None);
            device_context.PSSetShader(&pixel_shader, None);
            device_context.VSSetConstantBuffers(0, Some(&[Some(constant_buffer.clone())]));
            device_context.OMSetBlendState(&blend_state, None, 0xffffffff);
            device_context.RSSetState(&rasterizer_state);
            device_context.OMSetDepthStencilState(&depth_stencil_state, 0);
            device_context.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            let stride = mem::size_of::<Vertex>() as u32;
            let offset = 0u32;
            device_context.IASetVertexBuffers(
                0,
                1,
                Some(&Some(vertex_buffer.clone())),
                Some(&stride),
                Some(&offset),
            );

            // Draw main triangle
            device_context.DrawInstanced(3, VIEW_COUNT, 0, 0);

            // Get hand positions before releasing swapchain image
            session.sync_actions(&[(&action_set).into()]).unwrap();

            let right_location = right_space
                .locate(&stage, xr_frame_state.predicted_display_time)
                .unwrap();
            let left_location = left_space
                .locate(&stage, xr_frame_state.predicted_display_time)
                .unwrap();

            // Draw hand triangles
            if left_action.is_active(&session, xr::Path::NULL).unwrap()
                && left_location
                    .location_flags
                    .contains(xr::SpaceLocationFlags::POSITION_VALID)
            {
                let mut hand_mapped = D3D11_MAPPED_SUBRESOURCE::default();
                device_context
                    .Map(
                        &constant_buffer,
                        0,
                        D3D11_MAP_WRITE_DISCARD,
                        0,
                        Some(&mut hand_mapped),
                    )
                    .expect("Failed to map constant buffer");

                let transform_data = hand_mapped.pData as *mut TransformBuffer;
                (*transform_data).view_proj[0] = compute_view_proj_matrix(&views[0]);
                (*transform_data).view_proj[1] = compute_view_proj_matrix(&views[1]);
                (*transform_data).model = model_matrix_from_pose(
                    &left_location.pose.position,
                    &left_location.pose.orientation,
                    0.1, // Smaller scale for hand triangles
                );

                device_context.Unmap(&constant_buffer, 0);
                device_context.DrawInstanced(3, VIEW_COUNT, 0, 0);
            }

            if right_action.is_active(&session, xr::Path::NULL).unwrap()
                && right_location
                    .location_flags
                    .contains(xr::SpaceLocationFlags::POSITION_VALID)
            {
                let mut hand_mapped = D3D11_MAPPED_SUBRESOURCE::default();
                device_context
                    .Map(
                        &constant_buffer,
                        0,
                        D3D11_MAP_WRITE_DISCARD,
                        0,
                        Some(&mut hand_mapped),
                    )
                    .expect("Failed to map constant buffer");

                let transform_data = hand_mapped.pData as *mut TransformBuffer;
                (*transform_data).view_proj[0] = compute_view_proj_matrix(&views[0]);
                (*transform_data).view_proj[1] = compute_view_proj_matrix(&views[1]);
                (*transform_data).model = model_matrix_from_pose(
                    &right_location.pose.position,
                    &right_location.pose.orientation,
                    0.1, // Smaller scale for hand triangles
                );

                device_context.Unmap(&constant_buffer, 0);
                device_context.DrawInstanced(3, VIEW_COUNT, 0, 0);
            }

            let mut printed = false;
            if left_action.is_active(&session, xr::Path::NULL).unwrap() {
                print!(
                    "Left Hand: ({:0<12},{:0<12},{:0<12}), ",
                    left_location.pose.position.x,
                    left_location.pose.position.y,
                    left_location.pose.position.z
                );
                printed = true;
            }

            if right_action.is_active(&session, xr::Path::NULL).unwrap() {
                print!(
                    "Right Hand: ({:0<12},{:0<12},{:0<12})",
                    right_location.pose.position.x,
                    right_location.pose.position.y,
                    right_location.pose.position.z
                );
                printed = true;
            }
            if printed {
                println!();
            }

            let rect = xr::Rect2Di {
                offset: xr::Offset2Di { x: 0, y: 0 },
                extent: xr::Extent2Di {
                    width: swapchain.resolution.0 as _,
                    height: swapchain.resolution.1 as _,
                },
            };

            swapchain.handle.release_image().unwrap();

            frame_stream
                .end(
                    xr_frame_state.predicted_display_time,
                    environment_blend_mode,
                    &[
                        &xr::CompositionLayerProjection::new().space(&stage).views(&[
                            xr::CompositionLayerProjectionView::new()
                                .pose(views[0].pose)
                                .fov(views[0].fov)
                                .sub_image(
                                    xr::SwapchainSubImage::new()
                                        .swapchain(&swapchain.handle)
                                        .image_array_index(0)
                                        .image_rect(rect),
                                ),
                            xr::CompositionLayerProjectionView::new()
                                .pose(views[1].pose)
                                .fov(views[1].fov)
                                .sub_image(
                                    xr::SwapchainSubImage::new()
                                        .swapchain(&swapchain.handle)
                                        .image_array_index(1)
                                        .image_rect(rect),
                                ),
                        ]),
                    ],
                )
                .unwrap();
        }
    }

    println!("exiting cleanly");
}

const COLOR_FORMAT: u32 = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB.0 as u32;
const VIEW_COUNT: u32 = 2;
const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;

struct Swapchain {
    handle: xr::Swapchain<xr::D3D11>,
    resolution: (u32, u32),
    render_target_views: Vec<ID3D11RenderTargetView>,
}

unsafe fn compile_shader(source: &str, entry_point: &str, target: &str) -> ID3DBlob {
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error_blob: Option<ID3DBlob> = None;

        let source_bytes = source.as_bytes();
        let entry_point_cstr = std::ffi::CString::new(entry_point).unwrap();
        let target_cstr = std::ffi::CString::new(target).unwrap();

        let hr = D3DCompile(
            source_bytes.as_ptr() as *const _,
            source_bytes.len(),
            None,
            None,
            None,
            PCSTR(entry_point_cstr.as_ptr() as *const u8),
            PCSTR(target_cstr.as_ptr() as *const u8),
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut blob,
            Some(&mut error_blob),
        );

        if hr.is_err() {
            if let Some(error_blob) = error_blob {
                let error_msg = std::slice::from_raw_parts(
                    error_blob.GetBufferPointer() as *const u8,
                    error_blob.GetBufferSize(),
                );
                let error_str = String::from_utf8_lossy(error_msg);
                panic!("Shader compilation failed: {}", error_str);
            } else {
                panic!("Shader compilation failed with no error message");
            }
        }

        blob.unwrap()
    }
}

// Helper function to compute view-projection matrix from OpenXR view
fn compute_view_proj_matrix(view: &xr::View) -> [f32; 16] {
    // Construct view matrix from pose
    let pose = &view.pose;
    let q = &pose.orientation;
    let p = &pose.position;

    // Convert quaternion to rotation matrix
    let xx = q.x * q.x;
    let yy = q.y * q.y;
    let zz = q.z * q.z;
    let xy = q.x * q.y;
    let xz = q.x * q.z;
    let yz = q.y * q.z;
    let wx = q.w * q.x;
    let wy = q.w * q.y;
    let wz = q.w * q.z;

    // Build rotation matrix from quaternion
    let r00 = 1.0 - 2.0 * (yy + zz);
    let r01 = 2.0 * (xy - wz);
    let r02 = 2.0 * (xz + wy);
    let r10 = 2.0 * (xy + wz);
    let r11 = 1.0 - 2.0 * (xx + zz);
    let r12 = 2.0 * (yz - wx);
    let r20 = 2.0 * (xz - wy);
    let r21 = 2.0 * (yz + wx);
    let r22 = 1.0 - 2.0 * (xx + yy);

    // View matrix is inverse of camera transform: [R^T | -R^T * t]
    // Transpose the rotation and transform the translation
    let tx = -(r00 * p.x + r10 * p.y + r20 * p.z);
    let ty = -(r01 * p.x + r11 * p.y + r21 * p.z);
    let tz = -(r02 * p.x + r12 * p.y + r22 * p.z);

    let view_matrix = [
        r00, r01, r02, 0.0, r10, r11, r12, 0.0, r20, r21, r22, 0.0, tx, ty, tz, 1.0,
    ];

    // Construct projection matrix from FOV
    let fov = &view.fov;
    let tan_left = fov.angle_left.tan();
    let tan_right = fov.angle_right.tan();
    let tan_up = fov.angle_up.tan();
    let tan_down = fov.angle_down.tan();

    let near = 0.1;
    let far = 100.0;

    let proj_matrix = [
        2.0 / (tan_right - tan_left),
        0.0,
        0.0,
        0.0,
        0.0,
        2.0 / (tan_up - tan_down),
        0.0,
        0.0,
        (tan_right + tan_left) / (tan_right - tan_left),
        (tan_up + tan_down) / (tan_up - tan_down),
        -far / (far - near),
        -1.0,
        0.0,
        0.0,
        -(far * near) / (far - near),
        0.0,
    ];

    // Multiply view * proj
    multiply_matrices(&view_matrix, &proj_matrix)
}

// Helper function to compute model matrix (translation + rotation)
fn compute_model_matrix(rotation_angle: f32) -> [f32; 16] {
    let cos_a = rotation_angle.cos();
    let sin_a = rotation_angle.sin();

    // Rotation around Y axis, translated 3 meters forward (negative Z in OpenXR)
    [
        cos_a, 0.0, sin_a, 0.0, 0.0, 1.0, 0.0, 0.0, -sin_a, 0.0, cos_a, 0.0, 0.0, 1.5, -3.0,
        1.0, // Position: 3m forward, 1.5m up from floor
    ]
}

// Helper to multiply two 4x4 matrices
fn multiply_matrices(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut result = [0.0f32; 16];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            }
        }
    }
    result
}
