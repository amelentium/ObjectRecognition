using Microsoft.AspNetCore.Components;

namespace ObjectRecognition.Components.Dialogs
{
    public class DialogBase : ComponentBase
    {
        protected bool IsOpen = false;

        public void Show()
        {
            IsOpen = true;
        }

        protected void Close()
        {
            IsOpen = false;
        }
    }

    public class Dialog : DialogBase
    {
        [Parameter]
        public EventCallback OnConfirm { get; set; }

        protected void Confirm(object arg = null)
        {
            OnConfirm.InvokeAsync(arg);
            IsOpen = false;
        }
    }

    public class Dialog<T> : DialogBase
    {
        [Parameter]
        public EventCallback<T> OnConfirm { get; set; }

        protected void Confirm(T arg)
        {
            OnConfirm.InvokeAsync(arg);
            IsOpen = false;
        }
    }
}
