//! Illustrates rendering using D3D11. Supports Windows with D3D11 capable hardware.
//!
//! Renders a spinning RGB triangle 1 meter in front of the user.
//!
//! This code has been designed to clearly separate OpenXR and graphics concerns.
//! Look for "INTERFACE POINT" comments to see where OpenXR and graphics systems interact.

use std::mem;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use glam::{Mat4, Quat, Vec3};
use openxr as xr;
use windows::{
    Win32::Foundation::*, Win32::Graphics::Direct3D::Fxc::*, Win32::Graphics::Direct3D::*,
    Win32::Graphics::Direct3D11::*, Win32::Graphics::Dxgi::Common::*, core::*,
};

// ============================================================================
// CONSTANTS
// ============================================================================

const COLOR_FORMAT: u32 = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB.0 as u32;
const VIEW_COUNT: u32 = 2;
const VIEW_TYPE: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;

// ============================================================================
// MAIN
// ============================================================================

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

    // Initialize OpenXR instance and system
    let (xr_instance, system, environment_blend_mode) = init_openxr_instance();

    unsafe {
        // Initialize D3D11 graphics
        let d3d = init_d3d11();
        let resources = create_render_resources(&d3d.device);

        // Create OpenXR session (INTERFACE POINT: passes D3D device to OpenXR)
        let mut xr_ctx = create_openxr_session(&xr_instance, system, &d3d.device);

        // Setup input tracking
        let input = setup_input_tracking(&xr_instance, &xr_ctx.session);

        // Main loop state
        let mut swapchain: Option<OpenXrSwapchain> = None;
        let mut event_storage = xr::EventDataBuffer::new();
        let mut session_running = false;
        let start_time = std::time::Instant::now();

        'main_loop: loop {
            // Handle Ctrl+C
            if !running.load(Ordering::Relaxed) {
                println!("requesting exit");
                match xr_ctx.session.request_exit() {
                    Ok(()) => {}
                    Err(xr::sys::Result::ERROR_SESSION_NOT_RUNNING) => break,
                    Err(e) => panic!("{}", e),
                }
            }

            // Poll OpenXR events
            while let Some(event) = xr_instance.poll_event(&mut event_storage).unwrap() {
                use xr::Event::*;
                match event {
                    SessionStateChanged(e) => {
                        println!("entered state {:?}", e.state());
                        if handle_session_state(&xr_ctx.session, e.state(), &mut session_running) {
                            break 'main_loop;
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

            // OPENXR: Begin frame and check if we should render
            let xr_frame_state = xr_ctx.frame_wait.wait().unwrap();
            xr_ctx.frame_stream.begin().unwrap();

            if !xr_frame_state.should_render {
                xr_ctx
                    .frame_stream
                    .end(
                        xr_frame_state.predicted_display_time,
                        environment_blend_mode,
                        &[],
                    )
                    .unwrap();
                continue;
            }

            // OPENXR: Create swapchain on first render (INTERFACE POINT)
            let swapchain = swapchain.get_or_insert_with(|| {
                create_openxr_swapchain(&xr_instance, system, &xr_ctx.session, &d3d.device)
            });

            // OPENXR: Acquire swapchain image
            let image_index = swapchain.handle.acquire_image().unwrap();
            swapchain.handle.wait_image(xr::Duration::INFINITE).unwrap();

            // OPENXR: Get view transforms for this frame
            let (_, views) = xr_ctx
                .session
                .locate_views(
                    VIEW_TYPE,
                    xr_frame_state.predicted_display_time,
                    &xr_ctx.stage,
                )
                .unwrap();

            // Calculate rotating model matrix
            let elapsed = start_time.elapsed().as_secs_f32();
            let rotation_angle = elapsed * 0.5;

            // GRAPHICS: Render the scene
            let rtv = &swapchain.render_target_views[image_index as usize];
            let clear_color = [0.0f32, 0.0, 0.0, 1.0];
            d3d.device_context.ClearRenderTargetView(rtv, &clear_color);
            d3d.device_context
                .OMSetRenderTargets(Some(&[Some(rtv.clone())]), None);

            setup_pipeline_state(&d3d.device_context, &resources, swapchain.resolution);

            // Draw main triangle
            update_transforms(
                &d3d.device_context,
                &resources.constant_buffer,
                &views,
                compute_model_matrix(rotation_angle),
            );
            d3d.device_context.DrawInstanced(3, VIEW_COUNT, 0, 0);

            // OPENXR: Get hand tracking data
            xr_ctx
                .session
                .sync_actions(&[(&input.action_set).into()])
                .unwrap();

            let right_location = input
                .right_space
                .locate(&xr_ctx.stage, xr_frame_state.predicted_display_time)
                .unwrap();
            let left_location = input
                .left_space
                .locate(&xr_ctx.stage, xr_frame_state.predicted_display_time)
                .unwrap();

            // Draw hand triangles
            if input
                .left_action
                .is_active(&xr_ctx.session, xr::Path::NULL)
                .unwrap()
                && left_location
                    .location_flags
                    .contains(xr::SpaceLocationFlags::POSITION_VALID)
            {
                update_transforms(
                    &d3d.device_context,
                    &resources.constant_buffer,
                    &views,
                    model_matrix_from_pose(
                        &left_location.pose.position,
                        &left_location.pose.orientation,
                        0.1,
                    ),
                );
                d3d.device_context.DrawInstanced(3, VIEW_COUNT, 0, 0);
            }

            if input
                .right_action
                .is_active(&xr_ctx.session, xr::Path::NULL)
                .unwrap()
                && right_location
                    .location_flags
                    .contains(xr::SpaceLocationFlags::POSITION_VALID)
            {
                update_transforms(
                    &d3d.device_context,
                    &resources.constant_buffer,
                    &views,
                    model_matrix_from_pose(
                        &right_location.pose.position,
                        &right_location.pose.orientation,
                        0.1,
                    ),
                );
                d3d.device_context.DrawInstanced(3, VIEW_COUNT, 0, 0);
            }

            // Print hand positions
            let mut printed = false;
            if input
                .left_action
                .is_active(&xr_ctx.session, xr::Path::NULL)
                .unwrap()
            {
                print!(
                    "Left Hand: ({:0<12},{:0<12},{:0<12}), ",
                    left_location.pose.position.x,
                    left_location.pose.position.y,
                    left_location.pose.position.z
                );
                printed = true;
            }

            if input
                .right_action
                .is_active(&xr_ctx.session, xr::Path::NULL)
                .unwrap()
            {
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

            // OPENXR: Release swapchain image
            let rect = xr::Rect2Di {
                offset: xr::Offset2Di { x: 0, y: 0 },
                extent: xr::Extent2Di {
                    width: swapchain.resolution.0 as _,
                    height: swapchain.resolution.1 as _,
                },
            };

            swapchain.handle.release_image().unwrap();

            // OPENXR: End frame and submit layers
            xr_ctx
                .frame_stream
                .end(
                    xr_frame_state.predicted_display_time,
                    environment_blend_mode,
                    &[&xr::CompositionLayerProjection::new()
                        .space(&xr_ctx.stage)
                        .views(&[
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
                        ])],
                )
                .unwrap();
        }
    }

    println!("exiting cleanly");
}

// ============================================================================
// HIGH-LEVEL INITIALIZATION (called directly from main)
// ============================================================================

/// Initialize OpenXR instance and system
fn init_openxr_instance() -> (xr::Instance, xr::SystemId, xr::EnvironmentBlendMode) {
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

    let instance = entry
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

    let instance_props = instance.properties().unwrap();
    println!(
        "loaded OpenXR runtime: {} {}",
        instance_props.runtime_name, instance_props.runtime_version
    );

    let system = instance
        .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .unwrap();

    let environment_blend_mode = instance
        .enumerate_environment_blend_modes(system, VIEW_TYPE)
        .unwrap()[0];

    (instance, system, environment_blend_mode)
}

/// Initialize Direct3D 11 device and context
unsafe fn init_d3d11() -> D3DContext {
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

        println!(
            "Created D3D11 device with feature level: {:?}",
            feature_level
        );

        D3DContext {
            device: device.unwrap(),
            device_context: device_context.unwrap(),
        }
    }
}

/// Create all graphics pipeline resources
unsafe fn create_render_resources(device: &ID3D11Device) -> RenderResources {
    unsafe {
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

        RenderResources {
            vertex_shader,
            pixel_shader,
            input_layout,
            vertex_buffer,
            constant_buffer,
            blend_state,
            rasterizer_state,
            depth_stencil_state,
        }
    }
}

/// Create OpenXR session with D3D11 device (INTERFACE POINT: OpenXR <-> Graphics)
fn create_openxr_session(
    instance: &xr::Instance,
    system: xr::SystemId,
    d3d_device: &ID3D11Device,
) -> OpenXrContext {
    let requirements = instance.graphics_requirements::<xr::D3D11>(system).unwrap();

    println!(
        "D3D11 min feature level: {:?}",
        requirements.min_feature_level
    );

    let (session, frame_wait, frame_stream) = unsafe {
        instance
            .create_session::<xr::D3D11>(
                system,
                &xr::d3d::SessionCreateInfoD3D11 {
                    device: d3d_device.as_raw() as *mut _,
                },
            )
            .unwrap()
    };

    let stage = session
        .create_reference_space(xr::ReferenceSpaceType::STAGE, xr::Posef::IDENTITY)
        .unwrap();

    OpenXrContext {
        _instance: instance.clone(),
        _system: system,
        session,
        frame_wait,
        frame_stream,
        stage,
    }
}

/// Setup OpenXR input tracking (actions and spaces)
fn setup_input_tracking(
    instance: &xr::Instance,
    session: &xr::Session<xr::D3D11>,
) -> InputTracking {
    let action_set = instance
        .create_action_set("input", "input pose information", 0)
        .unwrap();

    let right_action = action_set
        .create_action::<xr::Posef>("right_hand", "Right Hand Controller", &[])
        .unwrap();
    let left_action = action_set
        .create_action::<xr::Posef>("left_hand", "Left Hand Controller", &[])
        .unwrap();

    instance
        .suggest_interaction_profile_bindings(
            instance
                .string_to_path("/interaction_profiles/khr/simple_controller")
                .unwrap(),
            &[
                xr::Binding::new(
                    &right_action,
                    instance
                        .string_to_path("/user/hand/right/input/grip/pose")
                        .unwrap(),
                ),
                xr::Binding::new(
                    &left_action,
                    instance
                        .string_to_path("/user/hand/left/input/grip/pose")
                        .unwrap(),
                ),
            ],
        )
        .unwrap();

    session.attach_action_sets(&[&action_set]).unwrap();

    let right_space = right_action
        .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)
        .unwrap();
    let left_space = left_action
        .create_space(session, xr::Path::NULL, xr::Posef::IDENTITY)
        .unwrap();

    InputTracking {
        action_set,
        right_action,
        left_action,
        right_space,
        left_space,
    }
}

/// Handle OpenXR session state changes
fn handle_session_state(
    session: &xr::Session<xr::D3D11>,
    state: xr::SessionState,
    session_running: &mut bool,
) -> bool {
    match state {
        xr::SessionState::READY => {
            session.begin(VIEW_TYPE).unwrap();
            *session_running = true;
        }
        xr::SessionState::STOPPING => {
            session.end().unwrap();
            *session_running = false;
        }
        xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
            return true; // Exit main loop
        }
        _ => {}
    }
    false
}

/// Create OpenXR swapchain and render target views (INTERFACE POINT: OpenXR <-> Graphics)
unsafe fn create_openxr_swapchain(
    instance: &xr::Instance,
    system: xr::SystemId,
    session: &xr::Session<xr::D3D11>,
    d3d_device: &ID3D11Device,
) -> OpenXrSwapchain {
    let views = instance
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
        .map(|&texture_ptr| unsafe {
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
            d3d_device
                .CreateRenderTargetView(&texture, Some(&rtv_desc), Some(&mut rtv))
                .expect("Failed to create render target view");

            mem::forget(texture);
            rtv.unwrap()
        })
        .collect::<Vec<_>>();

    OpenXrSwapchain {
        handle,
        resolution,
        render_target_views,
    }
}

// ============================================================================
// MID-LEVEL HELPERS (rendering and pipeline management)
// ============================================================================

/// Update constant buffer with view-projection and model matrices
unsafe fn update_transforms(
    device_context: &ID3D11DeviceContext,
    constant_buffer: &ID3D11Buffer,
    views: &[xr::View],
    model_matrix: [f32; 16],
) {
    unsafe {
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        device_context
            .Map(
                constant_buffer,
                0,
                D3D11_MAP_WRITE_DISCARD,
                0,
                Some(&mut mapped),
            )
            .expect("Failed to map constant buffer");

        let transform_data = mapped.pData as *mut TransformBuffer;
        (*transform_data).view_proj[0] = compute_view_proj_matrix(&views[0]);
        (*transform_data).view_proj[1] = compute_view_proj_matrix(&views[1]);
        (*transform_data).model = model_matrix;

        device_context.Unmap(constant_buffer, 0);
    }
}

/// Set up graphics pipeline state for rendering
unsafe fn setup_pipeline_state(
    device_context: &ID3D11DeviceContext,
    resources: &RenderResources,
    swapchain_resolution: (u32, u32),
) {
    unsafe {
        let viewport = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: swapchain_resolution.0 as f32,
            Height: swapchain_resolution.1 as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        device_context.RSSetViewports(Some(&[viewport]));

        device_context.IASetInputLayout(&resources.input_layout);
        device_context.VSSetShader(&resources.vertex_shader, None);
        device_context.PSSetShader(&resources.pixel_shader, None);
        device_context.VSSetConstantBuffers(0, Some(&[Some(resources.constant_buffer.clone())]));
        device_context.OMSetBlendState(&resources.blend_state, None, 0xffffffff);
        device_context.RSSetState(&resources.rasterizer_state);
        device_context.OMSetDepthStencilState(&resources.depth_stencil_state, 0);
        device_context.IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        let stride = mem::size_of::<Vertex>() as u32;
        let offset = 0u32;
        device_context.IASetVertexBuffers(
            0,
            1,
            Some(&Some(resources.vertex_buffer.clone())),
            Some(&stride),
            Some(&offset),
        );
    }
}

/// Compile HLSL shader from source
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

// ============================================================================
// LOW-LEVEL UTILITIES (math and matrix operations)
// ============================================================================

/// Convert OpenXR Vector3f to glam Vec3
fn to_vec3(v: &xr::Vector3f) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// Convert OpenXR Quaternionf to glam Quat
fn to_quat(q: &xr::Quaternionf) -> Quat {
    Quat::from_xyzw(q.x, q.y, q.z, q.w)
}

/// Compute combined view-projection matrix from OpenXR view
fn compute_view_proj_matrix(view: &xr::View) -> [f32; 16] {
    let pose = &view.pose;

    // Create view matrix (inverse of camera transform)
    let rotation = to_quat(&pose.orientation);
    let position = to_vec3(&pose.position);
    let camera_transform = Mat4::from_rotation_translation(rotation, position);
    let view_matrix = camera_transform.inverse();

    // Construct projection matrix from asymmetric FOV
    let fov = &view.fov;
    let near = 0.1;
    let far = 100.0;

    // Convert FOV angles to frustum bounds at near plane
    let left = near * fov.angle_left.tan();
    let right = near * fov.angle_right.tan();
    let bottom = near * fov.angle_down.tan();
    let top = near * fov.angle_up.tan();

    let proj_matrix = Mat4::frustum_rh(left, right, bottom, top, near, far);

    // Multiply view and projection matrices
    (proj_matrix * view_matrix).to_cols_array()
}

/// Compute model matrix with rotation around Y axis
fn compute_model_matrix(rotation_angle: f32) -> [f32; 16] {
    let rotation = Quat::from_rotation_y(rotation_angle);
    let position = Vec3::new(0.0, 1.5, -3.0); // 3m forward, 1.5m up from floor
    Mat4::from_rotation_translation(rotation, position).to_cols_array()
}

/// Create model matrix from position, orientation, and scale
fn model_matrix_from_pose(pos: &xr::Vector3f, quat: &xr::Quaternionf, scale: f32) -> [f32; 16] {
    let rotation = to_quat(quat);
    let position = to_vec3(pos);
    let scale_vec = Vec3::splat(scale);
    Mat4::from_scale_rotation_translation(scale_vec, rotation, position).to_cols_array()
}

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

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

/// Direct3D 11 graphics context
struct D3DContext {
    device: ID3D11Device,
    device_context: ID3D11DeviceContext,
}

/// Graphics pipeline resources (shaders, buffers, states)
struct RenderResources {
    vertex_shader: ID3D11VertexShader,
    pixel_shader: ID3D11PixelShader,
    input_layout: ID3D11InputLayout,
    vertex_buffer: ID3D11Buffer,
    constant_buffer: ID3D11Buffer,
    blend_state: ID3D11BlendState,
    rasterizer_state: ID3D11RasterizerState,
    depth_stencil_state: ID3D11DepthStencilState,
}

/// OpenXR context (instance, system, session)
struct OpenXrContext {
    _instance: xr::Instance,
    _system: xr::SystemId,
    session: xr::Session<xr::D3D11>,
    frame_wait: xr::FrameWaiter,
    frame_stream: xr::FrameStream<xr::D3D11>,
    stage: xr::Space,
}

/// OpenXR input tracking (actions and spaces)
struct InputTracking {
    action_set: xr::ActionSet,
    right_action: xr::Action<xr::Posef>,
    left_action: xr::Action<xr::Posef>,
    right_space: xr::Space,
    left_space: xr::Space,
}

/// OpenXR swapchain with render target views
struct OpenXrSwapchain {
    handle: xr::Swapchain<xr::D3D11>,
    resolution: (u32, u32),
    render_target_views: Vec<ID3D11RenderTargetView>,
}
