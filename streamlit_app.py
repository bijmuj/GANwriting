import docx2txt
import streamlit as st
from PIL import Image as im

import postprocess
import preprocess
import utils

gen, device = utils.get_model()
gen.eval()


def handle_inputs(word_doc, image_file, ht_in, wd_in, space_in, indent_in, form):
    try:
        text = docx2txt.process(word_doc)
        image = im.open(image_file).convert("RGB")
        image_array = utils.convert_pic_to_mini_array(image)
        image_array = utils.filter_mini_array(image_array)

        with st.spinner("Processing..."):
            preprocessed_imgs = preprocess.preprocess_images(image_array)
            text_dataloader, spaces, indents, imgs_per_line = preprocess.preprocess_text(text)
            imgs = utils.convert_to_images(gen, text_dataloader, preprocessed_imgs, device)
            imgs = postprocess.crop_images(imgs)
            imgs = postprocess.write2canvas(
                imgs,
                spaces,
                indents,
                imgs_per_line,
                ht_in=ht_in,
                wd_in=wd_in,
                space_in=space_in,
                indent_in=indent_in,
            )
            new_imgs = []
            for i in imgs:
                new_imgs.append(
                    im.fromarray(i).convert("RGB")
                )  # converting each array to PIL Image objects

            new_imgs[0].save("./temp/out.pdf", save_all=True, append_images=new_imgs[1:])
            form.success("Done.")
    except Exception as e:
        form.write(e)


def main():
    st.set_page_config(layout="wide", page_title="GANWriting")
    st.title("GANwriting")
    input_form = st.form("Inputs:")
    col1, col2 = input_form.columns(2)

    with col1:
        st.header("Input files:")
        word_doc = st.file_uploader("Upload a word document:", type=["docx"])
        image_file = st.file_uploader(
            "Upload handwriting sample image:", type=["png", "jpeg", "jpg"]
        )

    with col2:
        st.header("Other parameters:")
        ht_in = st.slider("Top margin:", 0, 100, 50, 5)
        wd_in = st.slider("Left margin:", 0, 100, 50, 5)
        space_in = st.slider("Space width:", 10, 40, 20, 5)
        indent_in = st.slider("Indent width:", 10, 120, 80, 5)

    submit = input_form.form_submit_button(
        "Submit",
    )

    if submit:
        handle_inputs(word_doc, image_file, ht_in, wd_in, space_in, indent_in, input_form)
        st.download_button(
            "Download file",
            data=open("./temp/out.pdf", "rb"),
            file_name="out.pdf",
        )


main()
