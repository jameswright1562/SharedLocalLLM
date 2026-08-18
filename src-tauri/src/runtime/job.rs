use std::process::Child;

pub struct ProcessJob {
    #[cfg(windows)]
    raw: isize,
}

unsafe impl Send for ProcessJob {}
unsafe impl Sync for ProcessJob {}

impl ProcessJob {
    pub fn new() -> Option<Self> {
        #[cfg(windows)]
        {
            windows_job()
        }
        #[cfg(not(windows))]
        {
            Some(Self {})
        }
    }

    pub fn assign(&self, child: &Child) -> Result<(), crate::types::ErrorPayload> {
        #[cfg(windows)]
        {
            assign_windows(self.raw, child)
        }
        #[cfg(not(windows))]
        {
            let _ = child;
            Ok(())
        }
    }
}

#[cfg(windows)]
fn windows_job() -> Option<ProcessJob> {
    use windows::Win32::System::JobObjects::{
        CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    unsafe {
        let handle = CreateJobObjectW(None, None).ok()?;
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            std::ptr::from_ref(&info).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
        .ok()?;
        Some(ProcessJob {
            raw: handle.0 as isize,
        })
    }
}

#[cfg(windows)]
fn assign_windows(raw: isize, child: &Child) -> Result<(), crate::types::ErrorPayload> {
    use std::os::windows::io::AsRawHandle;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::JobObjects::AssignProcessToJobObject;
    unsafe {
        AssignProcessToJobObject(
            HANDLE(raw as *mut std::ffi::c_void),
            HANDLE(child.as_raw_handle()),
        )
        .map_err(|error| crate::types::ErrorPayload::new("job_assign", error.to_string(), None))
    }
}

impl Drop for ProcessJob {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(windows::Win32::Foundation::HANDLE(
                self.raw as *mut std::ffi::c_void,
            ));
        }
    }
}
