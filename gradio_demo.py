import gradio as gr

def greet(name):
    return f"Hello, {name}!"

iface = gr.Interface(fn=greet, inputs="text", 
                     outputs="text", 
                     title="Greeting App", 
                     description="Enter your name to get a greeting")

if __name__ == "__main__":
    iface.launch()