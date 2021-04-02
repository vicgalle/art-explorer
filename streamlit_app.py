import torch
import clip
from PIL import Image
import glob
import numpy as np
import streamlit as st


def main():
    st.title("Zero-shot semantic search for art paintings")
    """This demo demonstrates the use of zero-shot learning for returning the closest painting for a given text query."""

    # Read in models from the data files.
    model, preprocess, image_featuresl = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.title('Info')

    st.sidebar.write(
        """The database has around 86.400 paintings, so if your query is very specific, it will only return the closest painting to the text.
        """
    )

    st.sidebar.write(
        """Developed by @vicgalle.
        """
    )
    
    ex_names = [
        'a painting of a woman in a green dress',
        'a painting of two persons kissing on the bed',
        'a cubist painting of a flower',
        'a cubist painting of a smiling woman at a coffee shop',
        'a cubist painting with something made of wood in it',
        'a painting about death',
        'a picture of a fuel company',
        'a painting about a boxed commercial product',
        'a painting about a canned commercial product',
        'a painting of very fat people',
        'a painting with the dutch flag',
        'a painting with the spanish flag',
        'an expressionist painting of woman writing',
        'a painting about paintings',
        'a painting with 🦓',
        'a painting with 🤡',
        'a yellowish painting of a vase of flowers',
        'a woodblock print of mount fuji',
        'the cutest painting of all']
    example = st.selectbox(
        'Choose an example text from this selector', ex_names)

    inp = st.text_input('Or write your own query here!',
                        example, max_chars=1000)

    with st.beta_expander("Search options..."):
        topk = st.slider('Choose the number of results (sorted by closeness to query)', 1, 5, 1)
        radio = st.radio('Choose the art database', ('WikiArt', 'All'))
        
    st.subheader('Search results:')

    text = clip.tokenize(
        [inp]).to(device)

    text_features = model.encode_text(text)
    image_featuresl = image_featuresl.to(device)
    image_features = image_featuresl/image_featuresl.norm(dim=-1, keepdim=True).to(device)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features.float() @ text_features.float().T).softmax(dim=0)
    print(similarity.shape)
    values, indices = similarity[:, 0].topk(5)

    names = np.asarray(glob.glob("wikiart/*/*"))
    print(values)
    print(names[indices.cpu().numpy()])
    
    for i in range(topk):
        image = Image.open(names[indices.cpu().numpy()][i])
        st.image(image, width = 500)

    st.title("How does it compare to Google Image Search?")
    st.write('Despite our small database, the main differences start appearing when providing complex yet vague queries, such as "a cubist painting with something made of wood in it".')
    image = Image.open('google.png')
    st.write('These are first four results, only one of them satisfies the query:')
    st.image(image, use_column_width=True)

    st.header("The Great Image Analogizer!")

    img_file_buffer = st.file_uploader("Upload an image as your query", type=['png', 'jpg', 'jpeg'])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)

        st.image(
            image,
            caption=f"You amazing query image!",
            width = 500
        )

        image = preprocess(image).unsqueeze(0).to(device)
        query_features = model.encode_image(image)
        query_features = query_features/query_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features.float() @ query_features.float().T).softmax(dim=0)

        values, indices = similarity[:, 0].topk(3)

        names = np.asarray(glob.glob("wikiart/*/*"))
        print(values)
        print(names[indices.cpu().numpy()])

        st.subheader("Top 3 closest paintings to your query:")
        
        for i in range(3):
            image = Image.open(names[indices.cpu().numpy()][i])
            st.image(image, width = 500)
        




@st.cache(allow_output_mutation=True)
def load_model():
    """
    Create the torch models and load data.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    image_featuresl = torch.load('wikiart/image_features.pt')

    return model, preprocess, image_featuresl


if __name__ == "__main__":
    main()
