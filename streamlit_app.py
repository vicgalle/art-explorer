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
    model, image_featuresl = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
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

    image = Image.open(names[indices.cpu().numpy()][0])
    st.image(image, use_column_width=True)

    st.title("How does it compare to Google Image Search?")
    st.write('Despite our small database, the main differences start appearing when providing complex yet vague queries, such as "a cubist painting with something made of wood in it".')
    image = Image.open('google.png')
    st.write('These are first four results, only one of them satisfies the query:')
    st.image(image, use_column_width=True)

# Ensure that load_pg_gan_model is called only once, when the app first loads.


@st.cache(allow_output_mutation=True)
def load_model():
    """
    Create the tensorflow session.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_featuresl = torch.load('image_features.pt')

    return model, image_featuresl
# Ensure that load_tl_gan_model is called only once, when the app first loads.


if __name__ == "__main__":
    main()
