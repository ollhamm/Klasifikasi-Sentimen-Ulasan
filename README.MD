# Application Running Instructions

This application consists of a backend written using Flask and a frontend using Next.js. Here are the steps to run this application:

## Preparation

Make sure you have Python, Node.js, and npm installed on your computer before running this application.

## Package Installation

1. Open a terminal or command prompt.
2. Go to the `backend` directory.
3. Run the following command to install the Flask package and its dependencies:

    ``bash
    pip install -r requirements.txt
    ```

4. Return to the application's main directory.
5. Go to the `frontend` directory.
6. Run the following command to install the packages needed for the frontend:

    ``bash
    npm install
    ```

## Running the Backend

1. Return to the main directory of the application.
2. Go to the `backend` directory.
3. Run the following command to start the Flask backend:

    ``bash
    python src/app.py
    ```

## Running the Frontend

1. Return to the main directory of the application.
2. Go to the `frontend` directory.
3. Run the following command to run the Next.js frontend:

    ```bash
    npm run dev
    ```

## Accessing the Application

Once the backend and frontend are running, you can access the application through a browser by going to `http://localhost:3000`.
