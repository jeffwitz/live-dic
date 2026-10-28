//! Native Rust host for the shared DIC-RT WGSL kernels.

#[pyo3::pymodule]
mod _wgpu_native {
    use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use std::time::Instant;

    const ICGN_SHADER: &str = include_str!("../../python/src/lucas_kanade_icgn/shaders/icgn.wgsl");

    fn runtime_error(error: impl std::fmt::Display) -> PyErr {
        PyRuntimeError::new_err(error.to_string())
    }

    /// Minimal persistent wgpu host used to qualify the PyO3 boundary.
    #[pyclass]
    pub struct NativeRuntime {
        #[allow(dead_code)]
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    }

    #[pymethods]
    impl NativeRuntime {
        #[new]
        fn new() -> PyResult<Self> {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                }))
                .map_err(runtime_error)?;
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("DIC-RT PyO3 native device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                }))
                .map_err(runtime_error)?;

            // Compile the same source consumed by wgpu-py. Creating the module
            // here catches WGSL drift before the Rust host grows its pipelines.
            let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("DIC-RT shared IC-GN kernels"),
                source: wgpu::ShaderSource::Wgsl(ICGN_SHADER.into()),
            });
            if let Some(error) = pollster::block_on(error_scope.pop()) {
                return Err(runtime_error(error));
            }

            Ok(Self {
                instance,
                adapter,
                device,
                queue,
            })
        }

        /// Return the selected native adapter as a stable Python dictionary.
        fn adapter_info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let info = self.adapter.get_info();
            let dictionary = pyo3::types::PyDict::new(py);
            dictionary.set_item("name", info.name)?;
            dictionary.set_item("vendor", info.vendor)?;
            dictionary.set_item("device", info.device)?;
            dictionary.set_item("device_type", format!("{:?}", info.device_type))?;
            dictionary.set_item("backend", format!("{:?}", info.backend))?;
            Ok(dictionary.into_any().unbind())
        }

        /// Measure direct borrowing and texture upload of a contiguous uint8 frame.
        ///
        /// ``enqueue_ms`` measures ``Queue::write_texture`` after PyO3 has
        /// validated and borrowed the array. ``synchronized_ms`` additionally
        /// waits for upload completion and is the transfer-latency boundary.
        fn benchmark_upload(
            &self,
            py: Python<'_>,
            frame: PyReadonlyArray2<'_, u8>,
            repeats: usize,
        ) -> PyResult<Py<PyAny>> {
            if repeats == 0 {
                return Err(PyValueError::new_err("repeats must be greater than zero"));
            }
            if !frame.is_c_contiguous() {
                return Err(PyValueError::new_err("frame must be C-contiguous"));
            }
            let bytes = frame.as_slice()?;
            let shape = frame.shape();
            let height = u32::try_from(shape[0]).map_err(runtime_error)?;
            let width = u32::try_from(shape[1]).map_err(runtime_error)?;
            let extent = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("DIC-RT PyO3 upload probe"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let destination = wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            };
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            };

            // Ensure one-time driver allocation is outside the reported samples.
            self.queue.write_texture(destination, bytes, layout, extent);
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(runtime_error)?;

            let mut enqueue_samples = Vec::with_capacity(repeats);
            let mut synchronized_samples = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let start = Instant::now();
                self.queue.write_texture(destination, bytes, layout, extent);
                enqueue_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
                self.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .map_err(runtime_error)?;
                synchronized_samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            }
            let result = pyo3::types::PyDict::new(py);
            result.set_item("enqueue_ms", enqueue_samples)?;
            result.set_item("synchronized_ms", synchronized_samples)?;
            Ok(result.into_any().unbind())
        }

        /// Expose the embedded shader size for packaging/non-duplication tests.
        fn shader_size(&self) -> usize {
            ICGN_SHADER.len()
        }
    }
}
