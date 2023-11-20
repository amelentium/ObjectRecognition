using Microsoft.AspNetCore.Components;

namespace ObjectRecognition.Components.Dialogs
{
    public class DialogBase : ComponentBase
    {
        protected bool IsOpen = false;

        protected void ShowDialog()
        {
            IsOpen = true;
        }

        protected void CloseDialog()
        {
            IsOpen = false;
        }

        public void Show()
        {
            ShowDialog();
        }
    }

    public class Dialog : DialogBase
    {
        [Parameter]
        public EventCallback OnConfirm { get; set; }

        protected void ConfirmAction(object arg = null)
        {
            OnConfirm.InvokeAsync(arg);
            IsOpen = false;
        }
    }

    public class Dialog<T> : DialogBase
    {
        [Parameter]
        public EventCallback<T> OnConfirm { get; set; }

        protected void ConfirmAction(T arg)
        {
            OnConfirm.InvokeAsync(arg);
            IsOpen = false;
        }
    }
}
